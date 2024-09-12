'''
This is the class of custom ML regression models based on Scikit-learn. 

'''
import warnings
warnings.filterwarnings("ignore")

import time
import copy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

## custom modules
import ML_Tools_Evaluation    ## sys.path.append('./')

#########################################################################################
############################### Regression model ########################################
#########################################################################################
class Regression_Model(object):
    ## <----- model initiation ---->
    def __init__(self,  myScikitModel=None, modelName='ML_Model', random_state=66, n_jobs=-1):     
        assert myScikitModel is None, f"Warning! Please define a model, Got: {myScikitModel}"
        self._name = modelName
        self._rng = random_state
        self._n_jobs = n_jobs
        self.model = myScikitModel
        self.performance = {}
        self.SearchResults = {}
        print(self.model)
            
    ## <----- hyper Parameter Search ---->
    def HyperParamSearch(self, search_space, X_train, y_train, search_method='Bayes', scoring='neg_mean_absolute_error', nFolds=5):
        ## define the searching space of hyperparameters
        #search_space = {"gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], "C": [1, 10, 100, 1e3, 1e4, 1e5]}

        ## count time
        beginTime = time.time()
        ## --------------------------------
        self.SearchResults[search_method] = {}

        ## initiate the search model
        if search_method == 'Bayes':
            optimizer = BayesSearchCV(estimator=self.model, search_spaces=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs, n_iter=100)
        else:
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        
        ## fit the Optimizer to the Data
        optimizer.fit(X_train, y_train)

        ## search results
        self.SearchResults[search_method]['best_model'] = best_model = optimizer.best_estimator_
        self.SearchResults[search_method]['best_score'] = best_score = optimizer.best_score_
        self.SearchResults[search_method]['best_param'] = best_param = best_model.get_params()
        ## --------------------------------
        print("Hyperparameter searching (Grid) costs time = %ds ................" % (time.time()-beginTime))
        print("Best %s is: %.4f" % (scoring, best_score))
        print("Best hyperparameters:", best_param)
        return optimizer
    
    def UpdateModel(self, ParamDict=None):
        if ParamDict:
            self.model.set_params(**ParamDict)
        print(self.model)
        
    ## <=============================== model train&validation/test ===============================>  
    def BuildModel(self, X_train, y_train, X_test, y_test, plot=True, labels={'x':'Pred', 'y':'Exp'}, colorby=None, trainingSetAnnotation='Training'):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        ## fit the model
        self.model.fit(X_train, y_train)
        ## make prediction
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        ## calculate the statistics and save
        self.performance[trainingSetAnnotation] = ML_Tools_Evaluation.CalcScores(y_train_pred, y_train)
        self.performance['Test'] = ML_Tools_Evaluation.CalcScores(y_test_pred, y_test)

        ## plot the predicted value VS the real value
        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5),  sharex=False, sharey=False)
            plt.subplots_adjust(wspace=0.5, hspace=0.1)
            axs[0] = ML_Tools_Evaluation.PlotPrediction(axs[0], y_train_pred, y_train, colorby=colorby, labels=labels, title=trainingSetAnnotation)
            axs[1] = ML_Tools_Evaluation.PlotPrediction(axs[1], y_test_pred, y_test, colorby=colorby, labels=labels, title='Test')
        ## --------------------------------
        print("Model construction costs time = %ds ................" % (time.time()-beginTime))
        return None
    
    ## <=============================== cross validation evaluation ===============================>
    def BuildModel_CV(self, X, y, nFolds=5, shuffle=True):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        CV_results = {}
        
        # define evaluation procedure
        cv = StratifiedKFold(n_splits=nFolds, shuffle=shuffle, random_state=self.rng)
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=self.rng)

        # evaluate model
        fold_id = 0
        for train, val in cv.split(X, y):
            ## fit the model
            self.model.fit(X[train], y[train])
            ## make prediction
            y_pred = self.model.predict(X[val])
            ## evaluate the performance
            print(f">>>> CV_{fold_id}")
            CV_results[fold_id] = ML_Tools_Evaluation.CalcScores(y_pred=y_pred, y_true=y[val])
            fold_id += 1

        ## save the results
        self.performance[f'CrossValidation({cv}-fold)'] = ML_Tools_Evaluation.CalcScoresStats_CV(CV_results)
        ## --------------------------------
        print("Cross validation costs time = %ds ................" % (time.time()-beginTime))
        print('Mean MAE (sklearn): %.3f (+/-%.3f)' % (np.mean(scores)*-1, np.std(scores)))
        return CV_results
#########################################################################################
############################# Regression model done #####################################
#########################################################################################
    
def makePrediction_logD_v1(dataTable, myModel, colName_model):
    dataTable_new = copy.deepcopy(dataTable)
    
    try:
        w, b = myModel.coef_[0][0], myModel.intercept_[0]
    except Exception as e:
        print(f"Error! Please check the model: {e}")
    else:
        colName_Cxlogd = "ChemAxon_logD[pH=7.4]"    # "logD[pH=7.4]"
        if colName_Cxlogd in dataTable_new.columns:
            dataTable_new[colName_model] = w * dataTable_new[colName_Cxlogd] + b
        else:
            print(f"Error! Please check the column name for ChemAxon logD: {colName_Cxlogd}")
    return dataTable_new


#########################################################################################
############################### Classification model ####################################
#########################################################################################
class Classification_Model(object):
    ## <----- model initiation ---->
    def __init__(self,  myScikitModel, modelName='ML_Model', random_state=66, n_jobs=-1):     
        #assert not myScikitModel, f"Warning! Please define a model, Got: {myScikitModel}"
        self._name = modelName
        self._rng = random_state
        self._n_jobs = n_jobs
        self.model = myScikitModel
        self.performance = {}
        self.SearchResults = {}
        print(self.model)
            
    ## <----- hyper Parameter Search ---->
    def HyperParamSearch(self, search_space, X_train, y_train, search_method='Bayes', scoring='neg_mean_absolute_error', nFolds=5):
        ## define the searching space of hyperparameters
        #search_space = {"gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], "C": [1, 10, 100, 1e3, 1e4, 1e5]}

        ## count time
        beginTime = time.time()
        ## --------------------------------
        self.SearchResults[search_method] = {}

        ## initiate the search model
        if search_method == 'Bayes':
            optimizer = BayesSearchCV(estimator=self.model, search_spaces=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs, n_iter=100)
        else:
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        
        ## fit the Optimizer to the Data
        optimizer.fit(X_train, y_train)

        ## search results
        self.SearchResults[search_method]['best_model'] = best_model = optimizer.best_estimator_
        self.SearchResults[search_method]['best_score'] = best_score = optimizer.best_score_
        self.SearchResults[search_method]['best_param'] = best_param = best_model.get_params()
        ## --------------------------------
        print("Hyperparameter searching (Grid) costs time = %ds ................" % (time.time()-beginTime))
        print("Best %s is: %.4f" % (scoring, best_score))
        print("Best hyperparameters:", best_param)
        return optimizer
    
    def UpdateModel(self, ParamDict=None):
        if ParamDict:
            self.model.set_params(**ParamDict)
        print(self.model)
        
    ## <=============================== model train&validation/test ===============================>  
    def TrainModel(self, X_train, y_train, X_test, y_test, trainingSetAnnotation='Training'):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        ## fit the model
        self.model.fit(X_train, y_train)
        ## make prediction
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        ## calculate the statistics and save
        self.performance[trainingSetAnnotation] = ML_Tools_Evaluation.CalcScores(y_train_pred, y_train)
        self.performance['Test'] = ML_Tools_Evaluation.CalcScores(y_test_pred, y_test)

        ## --------------------------------
        print("Model construction costs time = %ds ................" % (time.time()-beginTime))
        return None
    
    ## <=============================== cross validation evaluation ===============================>
    def TrainModel_CV(self, X, y, n_splits=5, shuffle=True):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        if shuffle:
            random_state = self._rng
        else:
            random_state = None
        
        ## run cross validation and check AUC
        fig, CV_results = ML_Tools_Evaluation.RocAUC_CrossValidation(
            classifier=self.model,
            X=X, 
            y=y, 
            n_splits=n_splits, 
            random_state=random_state)

        self.performance[f'cv_{n_splits}'] = CV_results
        ## --------------------------------
        print("Cross validation costs time = %ds ................" % (time.time()-beginTime))
        return fig, CV_results
#########################################################################################
############################# Regression model done #####################################
#########################################################################################