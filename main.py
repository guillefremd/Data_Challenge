from data_explorer import DataExplorer
from data_loader import DataLoader
from predictor import Predictor


def main(filename):
    #Loading data
    dataset = DataLoader(filename)
    explorer = DataExplorer(dataset)
    predictor = Predictor(dataset)
   
    #Exploring data
    explorer.visualise_distributions(without_outliers=False)

    explorer.visualise_distributions(without_outliers=True,identify_abnormal=True)
    explorer.describe_variable(652)
    explorer.histogram(652, without_outliers=True)
    explorer.boxplot(652, without_outliers=True)
        
    #PCA
    explorer.pca(plot=True)
    
    #Correlations
    explorer.best_relationship_class()
    explorer.visualise_best_relationship_class()
    explorer.best_relationship_pca()
    explorer.visualise_best_relationship_pca()

    #SVM
    a = Predictor(dataset)
    a.svc()

if __name__ == '__main__':
    main('dataset_challenge_one.tsv')
