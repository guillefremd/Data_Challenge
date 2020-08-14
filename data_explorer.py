import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
np.warnings.filterwarnings('ignore')


class DataExplorer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.describe = [
            stats.describe(variable_idx, nan_policy='omit')
            for variable_idx in data_loader.cleaned.T]
        self.describe_no_outliers = [
            stats.describe(variable_idx, nan_policy='omit')
            for variable_idx in data_loader.no_outliers.T
        ]

    def get_variable_data(self, variable_idx=None, variable_name=None,
                          without_outliers=False):
        if (variable_idx is not None and variable_name is not None) or \
           (variable_idx is None and variable_name is None):
            raise Exception(
                'Should specify one of variable_idx and variable_name')

        if variable_name is not None:
            variable_idx = self.data_loader.get_variable_idx(variable_name)

        if without_outliers:
            variable_data = self.data_loader.no_outliers.T[variable_idx]
        else:
            variable_data = self.data_loader.cleaned.T[variable_idx]

        return variable_data

    def histogram(self, variable_idx, without_outliers=False):
        variable_data = self.get_variable_data(
            variable_idx=variable_idx, without_outliers=without_outliers
        )
        if without_outliers:
            outliers_msg = '(outliers were removed)'
        else:
            outliers_msg = '(outliers were not removed)'

        plt.hist(
            variable_data, 25, color='c', edgecolor='k', alpha=0.35,
            weights=np.ones(len(variable_data)) / len(variable_data)
        )
        plt.axvline(
            np.nanmedian(variable_data), color='g', linestyle='dashed',
            linewidth=1
        )
        min_ylim, max_ylim = plt.ylim()
        plt.text(
            np.nanmedian(variable_data), max_ylim*0.95,
            'Median = {}. {}'.format(
                str(np.nanmedian(variable_data)), outliers_msg))
        plt.ylabel('Frequency (%)')
        plt.xlabel('Variable value')
        plt.title('Histogram of : {}'.format(
            self.data_loader.get_variable_name(variable_idx)))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()

    def boxplot(self, variable_idx, without_outliers=False):
        variable_data = self.get_variable_data(
            variable_idx=variable_idx, without_outliers=without_outliers
        )
        if without_outliers:
            outliers_msg = '(outliers were removed)'
        else:
            outliers_msg = '(outliers were not removed)'

        plt.boxplot(variable_data)
        plt.title('Boxplot of : {}'.format(
            self.data_loader.get_variable_name(variable_idx)))
        min_ylim, max_ylim = plt.ylim()
        plt.text(
            np.nanmedian(variable_data), max_ylim*0.95,
            'Median= {}. {}'.format(
                str(np.nanmedian(variable_data)), outliers_msg))
        plt.show()

    def visualise_distributions(self, without_outliers=False,
                                identify_abnormal=False):
        if without_outliers:
            description = self.describe_no_outliers
            outliers_msg = '(outliers were removed)'
        else:
            description = self.describe
            outliers_msg = '(outliers were not removed)'

        kurtosis = []
        skewness = []
        for each_variable in description:
            kurtosis.append(each_variable[4])
            skewness.append(each_variable[5])

        plt.ylabel('Kurtosis')
        plt.xlabel('Skewness')
        plt.scatter(skewness, kurtosis, alpha=0.8, color='g', marker='.')
        plt.title('Distribution of all variables: {}'.format(outliers_msg))

        if identify_abnormal:
            print('The following variables present abnormal distributions')
            for variable_idx, (k, s) in enumerate(zip(kurtosis, skewness)):
                if s > 1 or s < -1 or k > 1 or k < -1:
                    plt.annotate(
                        self.data_loader.get_variable_name(variable_idx),
                        (s, k), size=6, c='b')
                    print('{}: Skewness={}, Kurtosis={}'.format(
                        self.data_loader.get_variable_name(variable_idx),
                        str(s), str(k)))

        plt.show()

    def describe_variable(self, variable_idx):
        name = self.data_loader.get_variable_name(variable_idx)
        data = self.describe[variable_idx]
        print('Summary statistics of ' + name)
        print('Number of observations: ' + str(data[0]))
        print('Range: ' + str(data[1]))
        print('Mean: ' + str(data[2]))
        print('Variance: ' + str(data[3]))
        print('Standard deviation: ' + str(math.sqrt(data[3])))
        print('Skewness: ' + str(data[4]))
        print("Kurtosis: " + str(data[5]))

    def pca(self, plot=False):
        dataX = self.data_loader.scaled[:, 0:-1]
        dataY = self.data_loader.cleaned[:, -1]
        pca = PCA(n_components=2)
        pca.fit(dataX)
        X_pca = pca.transform(dataX)

        if not plot:
            return X_pca
        else:
            Xax = X_pca[:, 0]
            Yax = X_pca[:, 1]
            labels = dataY
            cdict = {min(dataY): 'red', max(dataY): 'green'}
            labl = {min(dataY): '0', max(dataY): '1'}
            marker = {min(dataY): 'o', max(dataY): 'x'}

            alpha = {min(dataY): .3, max(dataY): .8}
            fig, ax = plt.subplots(figsize=(7, 5))

            for label in np.unique(labels):
                ix = np.where(labels == label)
                ax.scatter(
                    Xax[ix], Yax[ix], s=40, label=labl[label],
                    marker=marker[label], alpha=alpha[label])

            plt.xlabel('First Principal Component', fontsize=14)
            plt.ylabel('Second Principal Component', fontsize=14)
            plt.title('PCA')
            plt.legend()
            plt.show()

    def best_relationship_class(self):
        label = self.data_loader.cleaned[:, -1]
        variables_data = self.data_loader.scaled[:, :-1]
        relationships_array = np.array([
            stats.pointbiserialr(variable, label)
            for variable in variables_data.T])
        variables_names = self.data_loader.columns[:-1]

        max_value = 0
        var = 0
        variable_max = []
        for i in relationships_array:
            if max_value == 0 or abs(i[0]) > max_value:
                max_value = i[0]
                variable_max = i
                best_var = var
            var += 1
        print(
            '{} presents a Point Biserial Correlation of {} with the Class'
            ' variable. (p-value={})'.format(
                variables_names[best_var], max_value, str(variable_max[1])))

        return (relationships_array,
                variable_max,
                best_var,
                variables_names[best_var])

    def visualise_best_relationship_class(self):
        details = self.best_relationship_class()
        best_variable_data = self.data_loader.np_array.T[details[2]]

        class_data = self.data_loader.np_array.T[-1]

        class_0 = best_variable_data[class_data == 0]
        class_1 = best_variable_data[class_data == 1]
        plt.ylabel(details[3])
        patch_artist = True

        plt.title('Relationship between Class and ' + details[3])
        boxp = plt.boxplot(
            [class_0, class_1], patch_artist=True,
            labels=['Class 0', 'Class 1'])
        colors = ['lightblue', 'lightgreen']

        for patch, color in zip(boxp['boxes'], colors):
            patch.set_facecolor(color)

        txt = (
            '\n{} presents a Point Biserial Correlation of {} with the Class'
            ' variable.\n (p-value={})'.format(
                details[3], str(details[1][0]), str(details[1][1])))
        plt.figtext(
            0.5, 0.01, txt, wrap=True, horizontalalignment='center',
            fontsize=8)

        plt.show()

    def best_relationship_pca(self):
        pca1_data = self.pca(plot=False)[:, 0]

        variables_data = self.data_loader.cleaned[:, :-1]
        relationships_array = np.array([
            stats.spearmanr(variable, pca1_data)
            for variable in variables_data.T])

        variables_names = self.data_loader.columns[:-1]

        max_value = 0
        var = 0
        variable_max = []
        for i in relationships_array:
            if max_value == 0 or abs(i[0]) > max_value:
                max_value = i[0]
                variable_max = i
                best_var = var
            var += 1

        print(
            '{} presents a Spearman correlation of {} with the first Principal '
            ' Component.\n (p-value={})'.format(
                variables_names[best_var], str(max_value),
                str(variable_max[1])))
        return (relationships_array,
                variable_max,
                best_var,
                variables_names[best_var],
                pca1_data)

    def visualise_best_relationship_pca(self):
        pca_details = self.best_relationship_pca()
        pca1_data = pca_details[4]
        variable_data = self.data_loader.cleaned.T[pca_details[2]]

        plt.scatter(variable_data, pca1_data, alpha=0.7)
        plt.ylabel('First Principal Component\n', fontsize=12)
        plt.xlabel(pca_details[3], fontsize=12)
        plt.title('Plot PC1 - ' + pca_details[3])
        txt = (
            '\n{} presents a Spearman correlation of {} with first Principal '
            ' Component.\n (p-value={})'.format(
                pca_details[3], str(np.around(pca_details[1][0], 2)),
                str(pca_details[1][1])))
        plt.figtext(
            0.5, 0.01, txt, wrap=True, horizontalalignment='center',
            fontsize=8)

        plt.show()
