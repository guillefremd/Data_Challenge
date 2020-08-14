from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Predictor:
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader
        self.data = dataset_loader.cleaned

    def svc(self):

        variables = self.data[:, : -1]
        labels = self.data[:, -1]
        variables_train, variables_test, labels_train, labels_test = train_test_split(variables, labels, test_size=0.20, random_state=0)

        parameter_candidates = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
           'kernel': ['rbf']}, ]

        clf = GridSearchCV(estimator=svm.SVC(),
                           param_grid=parameter_candidates, n_jobs=-1)

        clf.fit(variables_train, labels_train)

        print('Best score for training data: ', clf.best_score_)
        print('Best C:', clf.best_estimator_.C)
        print('Best Kernel:', clf.best_estimator_.kernel)
        print('Best Gamma:', clf.best_estimator_.gamma)

        variables_train, variables_test, labels_train, labels_test = train_test_split(variables, labels,
                                       test_size=0.30, random_state=0)
        clf=svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma)
        clf=clf.fit(variables_train, labels_train)
        prediction=clf.predict(variables_test)
        print()
        print(
            'The SVM classifier correctly predicted {} out of {}'
            'values.'.format(
                str(sum(prediction == labels_test)), str(len(prediction))))
        print('This represents an error rate of {:.2f}%'.format(
            sum(prediction != labels_test) / len(prediction) * 100
        ))


        title = "Confusion matrix - SVM Prediction"
    
        disp = plot_confusion_matrix(clf, variables_test, labels_test,
                                 display_labels=["0","1"],
                                 cmap=plt.cm.Blues
                                 )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        plt.show()
