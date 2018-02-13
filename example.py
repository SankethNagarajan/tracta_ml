from tracta_ml.model_optimizer import ModelTuner
from sklearn.model_selection import KFold
from datetime import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# including code in main block for parallelization to work
if __name__ == '__main__':

    # loading WDBC dataset from UCI repo - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    # Remove the ID column before feeding the dataset to this code

    test_df = pd.read_csv('breast_cancer_data.csv')

    mod = SVC(cache_size=1000, random_state=40)  # initializing model
    sc = StandardScaler()

    # defining parameter search space

    param_dict = {
        'C': ['float', 0, 2000],
        'coef0': ['float', 0, 1000],
        'degree': ['int', 2, 5],
        'gamma': ['float', 0, 10000],
        'kernel': ['list', 'linear', 'poly', 'rbf', 'sigmoid'],
        'shrinking': ['list', True, False]
    }

    X = test_df.loc[:, test_df.columns != 'diagnosis']
    Y = test_df.loc[:, 'diagnosis']
    Y = Y.apply(lambda x: 1 if x == 'M' else 0)
    kfold_cv = KFold(10, shuffle=True, random_state=15)

    # preprocessing input data for SVM
    X1 = pd.DataFrame(sc.fit_transform(X), columns=X.columns.values)

    ###################################################################################################

    tuner = ModelTuner(mod, param_dict, kfold_cv, 'accuracy')  # initializing optimizer

    tuner.fit(X1, Y, verbose=False)  # optimizing hyper-parameters and features

    best_model = tuner.get_best_model()  # retrieving best model
    best_features = tuner.get_features()  # retrieving best feature set

    X_trans = tuner.transform(X)  # transforming input according to best feature set

    # plotting optimization monitors

    tuner.plot_monitor('Model_Fitness')
    tuner.plot_monitor('Feature_Fitness')
    tuner.plot_monitor('Stdev')

    loaded_model = tuner.load_file('best_model.pkl') # loading the best model from disk

    print(loaded_model)

    ###########################END-OF-CODE#################################