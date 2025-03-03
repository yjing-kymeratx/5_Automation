'''
This is the class of custom ML regression/classification models based on Scikit-learn. 

'''
import warnings
warnings.filterwarnings("ignore")

import time
import copy
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##
from scipy.stats import linregress, t, pearsonr, spearmanr, kendalltau

# from skopt import BayesSearchCV
from sklearn.metrics import mean_absolute_error, roc_auc_score, auc, roc_curve, accuracy_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV

## custom modules

####################################################################
####################### data preprocessing #########################
####################################################################
class preProcessor(object):
    ## ================= initialize object =================
    def __init__(self):
        self.dict_stats_norm = {}
        self.transformType = 'None'
        self.dataDict_v2b, self.dataDict_b2v = {}, {}

    ## ================= preprossing X =================
    def PreProcess_X(self, dataTable_X, cols_sele=None, isTrain=True):
        ## -----------------------------------   
        if cols_sele is None:
            colName_tbp = []
        elif cols_sele == "All" or cols_sele == "*":
            colName_tbp = dataTable_X.columns
        else:
            colName_tbp = cols_sele

        ## -----------------------------------   
        X = copy.deepcopy(dataTable_X)
        for col in colName_tbp:
            if col not in X.columns:
                print(f'\tWarning! This column {col} is not in dataTable_X')
            else:
                if isTrain:
                    self.dict_stats_norm[col] = self._learn_norm(X[col].to_numpy())
                
                if col not in self.dict_stats_norm:
                    print(f'\tWarning! This column {col} is not in the trained stats dictionary')
                else:
                    X[col] = X[col].apply(lambda v: self._zscore_norm(v, self.dict_stats_norm[col]))                    
        return X            

    ## ================= preprossing y =================
    def PreProcess_y(self, dataTable_y, transformType='None', isTrain=True):
        self.transformType = transformType
        assert len(dataTable_y.columns) == 1, f"\tError! The <dataTable_y> has incorrect {len(y.columns)} columns: {y.columns}"
        y = copy.deepcopy(dataTable_y)
        colName_raw = dataTable_y.columns[0]
        # y = dataTable_y.rename(columns={dataTable_y.columns[0]: 'y_processed'})
        #         
        if self.transformType=='log10':
            ## process log value on y
            try:
                y['y_preprocess'] = y[colName_raw].apply(lambda x: np.log10(x))
            except Exception as e:
                y['y_preprocess'] = y[colName_raw].apply(lambda x: np.nan)

        elif self.transformType=='One-hot':
            if isTrain:
                y_values = sorted(y[colName_raw].unique())
                for i in range(len(y_values)):
                    self.dataDict_v2b[str(y_values[i])] = str(i)
                    self.dataDict_b2v[str(i)] = str(y_values[i])
                print(f"\tThere are total {len(self.dataDict_v2b)} classes: {self.dataDict_v2b}")
            try:
                y['y_preprocess'] = y[colName_raw].apply(lambda x: int(self.dataDict_v2b[str(x)]))
            except Exception as e:
                print(f"Warning, cannot preprocess y, Error msg: {e}")
                y['y_preprocess'] = y[colName_raw].apply(lambda x: np.nan)
        else:
            print(f"\tWarning! <modelType> should be either <log10> for continuous data or <One-hot> for binary data")
            y['y_preprocess'] = y[colName_raw].apply(lambda x: x)
        
        y = y.drop([colName_raw], axis=1)
        return y

    def PostProcess_y(self, dataTable_y):
        assert len(dataTable_y.columns) == 1, f"Error! The <dataTable_y> has incorrect {len(y.columns)} columns: {y.columns}"
        colName_raw = dataTable_y.columns[0]
        y = copy.deepcopy(dataTable_y)

        if self.transformType=='log10':
            y['y_postprocess'] = y[colName_raw].apply(lambda x: 10**(x))
        
        elif self.transformType=='One-hot':
            y['y_postprocess'] = y[colName_raw].apply(lambda x: dataDict_b2v(str(x)))
        else:
            y['y_postprocess'] = y[colName_raw].apply(lambda x: x)
        return y

    ## ================= define internal stats funcs ==============
    ## ------- define stats funcs -------
    def _learn_norm(self, np_list):
        dict_stats = {}
        try:
            np_list = np.array(np_list)
            dict_stats['max'] = np.max(np_list)
            dict_stats['min'] = np.min(np_list)
            dict_stats['median'] = np.median(np_list)
            dict_stats['mean'] = np.mean(np_list)
            dict_stats['std'] = np.std(np_list)
        except Exception as e:
            dict_stats['max'] = 10**9
            dict_stats['min'] = -10**9
            dict_stats['median'] = np.nan
            dict_stats['mean'] = 0
            dict_stats['std'] = 1
        return dict_stats

    def _zscore_norm(self, value, dict_stats):
        if value > dict_stats['max'] or value < dict_stats['min']:
            min, max = round(dict_stats['min'], 2), round(dict_stats['max'], 2)
            print(f"\tWarning! The value {value} is out of range ({min}, {max})")
        
        if dict_stats['std'] == 0:
            value_normed = value
        else:
            try:
                value_normed = (value - dict_stats['mean'])/dict_stats['std']
            except Exception as e:
                value_normed = value
        return value_normed
    ## ============================================================

#########################################################################################
############################# Regression model done #####################################
#########################################################################################
class Regression_Model(object):
    ## <===================== model initiation =====================>
    def __init__(self,  myScikitModel=None, modelName='Regression_Model', rng=666666, n_jobs=-1):
        assert myScikitModel is not None, f"\tWarning! Please define an initiated RDKit ML model"
        self._name = modelName
        self._rng = rng
        self._n_jobs = n_jobs
        self.model = myScikitModel
        self.HPT_Results = {}
        self.predictions = None
        self.performance = {}
        self.plots = {}
            
    ## <===================== model training =====================>
    def Train(self, X, y, printLog=True, HPT=False, search_space=None):
        ## count time
        beginTime = time.time()
        ## ----------------------------------------------------------------
        ## ------------ hyper parameter search ------------
        if HPT:
            self._HyperParamSearch(X, y, search_space=search_space, printLog=printLog)
        
        ## ------------ fit the model ------------
        self.model.fit(X, y)
       
        ## ----------------------------------------------------------------
        print(f"\tModel construction costs time = {(time.time()-beginTime):.2f} s ................")
        return None

    ## <===================== model evaluation =====================>
    def Evaluate(self, X, y, ds_label='TBD', printLog=True, plotResult=False):
        ## make prediction
        y_pred = self.model.predict(X)

        ## save prediction
        df_predictions = copy.deepcopy(y)
        df_predictions['Experiment'] = df_predictions[y.columns[0]]
        df_predictions['DataSet'] = ds_label
        df_predictions['Prediction'] = y_pred
        self.predictions = pd.concat([self.predictions, df_predictions]) if self.predictions is not None else df_predictions

        ## calcualte statistics
        print(f"\tEvaluation results of the {ds_label} dataset:")
        self.performance[ds_label] = self._CalcScores(y_pred=y_pred, y_true=y.to_numpy(), printLog=printLog)
        
        ## plotting
        if plotResult:
            self.plots[ds_label] = self._Plot_Pred_VS_Expt(dataTable=df_predictions,
                                                           label_x='Prediction', 
                                                           label_y='Experiment',
                                                           color_by='DataSet',
                                                           figTitle=f"Pred VS Expt ({ds_label})")
        return None
    
    ## <===================== HPTunning =====================>
    def _HyperParamSearch(self, X, y, search_space=None, search_method='grid', scoring='neg_mean_absolute_error', nFolds=5, printLog=True):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        print(f"\tStart Hyper-Parameter Tunning ...")
        SearchResults = {'best_model': None, 'best_score':None, 'best_param':None}
        
        ##
        if search_method == 'grid':
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        elif search_method =='Bayes':
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        else:
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)

        ## fit the Optimizer to the Data
        y_reshaped = y.to_numpy().reshape((len(y), ))
        optimizer.fit(X, y_reshaped)

        ## search results
        SearchResults['best_model'] = optimizer.best_estimator_
        SearchResults['best_score'] = optimizer.best_score_
        SearchResults['best_param'] = SearchResults['best_model'].get_params()
        self.HPT_Results[search_method] = SearchResults
        
        ##
        # self.model = optimizer.best_estimator_
        if SearchResults['best_param'] is not None:
            self.model.set_params(**SearchResults['best_param'])
        else:
            self.model = self.model

        if printLog:
            print(f"\tThis is the log info")
            print(f"\tThe best {scoring}: {SearchResults['best_score']}")
            print(f"\tThe optimized Params: {SearchResults['best_param']}")
            ## ----------------------------------------------------------------
            print(f"\tHyper-parameters Tunning costs time = {(time.time()-beginTime):.2f} s ................")
        return None
    
    ## <===================== tools =====================>
    def _CalcScores(self, y_pred, y_true, printLog=True):   
        dataDict_result = {}
        try:
            y_pred = y_pred.reshape((len(y_pred), ))
            y_true = y_true.reshape((len(y_true), ))
        except Exception as e:
            print(f"\tError! Cannot reformatting the y_pred and y_true when calculating the statistics")
        else:
            ## calculate the mean absolute error using Scikit learn
            try:
                dataDict_result['MAE'] = mean_absolute_error(y_true, y_pred)
            except:
                dataDict_result['MAE'] = np.nan
            
            ## calculate the PearsonCorrelationCoefficient
            try:
                pr_np = np.corrcoef(y_pred, y_true)[1, 0]
                dataDict_result['Pearson_R2'] = pr_np * pr_np
            except:
                dataDict_result['Pearson_R2'] = np.nan

            ## calculate the rank-order correlation (Spearman's rho)
            try:
                sr_sp, sp_sp = spearmanr(y_pred, y_true)[0], spearmanr(y_pred, y_true)[1]
                dataDict_result['Spearman_R2'] = sr_sp * sr_sp
            except:
                dataDict_result['Spearman_R2'], sp_sp = np.nan, np.nan
                        
            ## calculate the # Kendall's tau
            try:
                kr_sp, kp_sp = kendalltau(y_pred, y_true)[0] , kendalltau(y_pred, y_true)[1]
                dataDict_result['KendallTau_R2'] = kr_sp * kr_sp
            except:
                dataDict_result['KendallTau_R2'], kp_sp = np.nan, np.nan
             
            ## print out the results
            if printLog:
                print(f"\t\tData shape: y_pred {y_pred.shape}; y_true {y_true.shape}")
                print(f"\t\tMean absolute error: {dataDict_result['MAE']:.2f}")
                print(f"\t\tPearson-R2: {dataDict_result['Pearson_R2']:.2f}")
                print(f"\t\tSpearman-R2: {dataDict_result['Spearman_R2']:.2f} (p={sp_sp:.2f})")
                print(f"\t\tKendall-R2: {dataDict_result['KendallTau_R2']:.2f} (p={kp_sp:.2f})")
        return dataDict_result
    
    def _Plot_Pred_VS_Expt(self, dataTable, label_x='Prediction', label_y='Experiment', color_by=None, diagonal=True, sideHist=True, figTitle=None):
        x, y = dataTable[label_x], dataTable[label_y]
        ## --------- Start with a square Figure ---------
        fig = plt.figure(figsize=(8, 8))

        if sideHist:
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ## --------- add hist ---------
            if sideHist:
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                bins = 10
                ax_histx.hist(x, bins=bins)
                ax_histy.hist(y, bins=bins, orientation='horizontal')
            
                ax_histx.tick_params(axis="x", labelbottom=False)    # no x labels
                ax_histy.tick_params(axis="y", labelleft=False)    # no y labels

                ax_histx.tick_params(axis='both', which='major', labelsize=16)
                ax_histy.tick_params(axis='both', which='major', labelsize=16)
        else:
            ax = fig.add_subplot()
        
        ## --------- add plot ---------
        if color_by is None:
            ax.scatter(x, y, s=40, alpha=0.5, cmap='Spectral', marker='o')
        else:
            for i in sorted(dataTable[color_by].unique()):
                idx = dataTable[dataTable[color_by]==i].index.to_list()
                ax.scatter(x.loc[idx], y.loc[idx], s=40, alpha=0.5, cmap='Spectral', marker='o', label=i)
            ax.legend(loc="upper left", title=color_by)    #, bbox_to_anchor=(1.35, 0.5)
        
        ## figure label and title
        ax.set_xlabel(label_x, fontsize=16)
        ax.set_ylabel(label_y, fontsize=16)

        # now determine nice limits:
        # ax_max = np.ceil(max(np.max(x), np.max(y)))
        # ax_min = np.floor(min(np.min(x), np.min(y)))
        ax_max = max(np.max(x), np.max(y))
        ax_min = min(np.min(x), np.min(y))
        ax_addon = (ax_max - ax_min)/10
        ax_max = ax_max + ax_addon
        ax_min = ax_min - ax_addon
        ax.set_xlim([ax_min, ax_max])
        ax.set_ylim([ax_min, ax_max])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(alpha=0.75)

        if diagonal: 
            diagonalLine = ax.plot([ax_min, ax_max], [ax_min, ax_max], c='lightgray', linestyle='-')
            # fold1Line1 = ax.plot([ax_min+1, ax_max], [ax_min, ax_max-1], c='lightgray', linestyle='--')
            # fold1Line2 = ax.plot([ax_min, ax_max-1], [ax_min+1, ax_max], c='lightgray', linestyle='--')
        
        figTitle = f"Pred vs Expt)" if figTitle is None else figTitle
        fig.suptitle(figTitle, fontsize=24)
        return fig

    def ___futureFunctionsTBA():
        return None

#########################################################################################
############################### Classification model ####################################
#########################################################################################
class Classification_Model(object):
    ## <===================== model initiation =====================>
    def __init__(self, myScikitModel=None, modelName='Classification_Model', rng=666666, n_jobs=-1):
        assert myScikitModel is not None, f"\tWarning! Please define an initiated RDKit ML model"
        self._name = modelName
        self._rng = rng
        self._n_jobs = n_jobs
        self.model = myScikitModel
        self.HPT_Results = {}
        self.predictions = None
        self.performance = {}
        self.plots = {}
        self.best_threshold = 0.5
        self.class_label = {0: "0", 1: "0"}

    ## <===================== model training =====================>
    def Train(self, X, y, printLog=True, HPT=False, search_space=None):
        ## count time
        beginTime = time.time()
        ## ----------------------------------------------------------------
        ## ------------ hyper parameter search ------------
        if HPT:
            self._HyperParamSearch(X, y, search_space=search_space, printLog=printLog)
        
        ## ------------ fit the model ------------
        self.model.fit(X, y)
       
        ## ----------------------------------------------------------------
        print(f"\tModel construction costs time = {(time.time()-beginTime):.2f} s ................")
        return None

    ## <===================== model evaluation =====================>
    def Evaluate(self, X, y, ds_label='TBD', estCutoff=False, printLog=True, plotResult=False):
        ## make prediction
        # y_pred = self.model.predict(X)    #####################
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        
        ## calcualte statistics
        print(f"\tEvaluation results of the {ds_label} dataset:")
        self.performance[ds_label] = self._CalcScores(y_pred=y_pred_prob, y_true=y.to_numpy(), estTrsd=estCutoff, printLog=printLog)

        ## save prediction
        df_predictions = copy.deepcopy(y)
        df_predictions['Experiment'] = df_predictions[y.columns[0]]
        df_predictions['DataSet'] = ds_label
        df_predictions['Prob_1'] = y_pred_prob
        df_predictions['Prediction_'] = df_predictions['Prob_1'].apply(lambda x: self._Pred_Class(x))
        self.predictions = pd.concat([self.predictions, df_predictions]) if self.predictions is not None else df_predictions
        
        ## plotting
        if plotResult:
            self.plots[ds_label] = self._Plot_ROCAUC(ds_label)
        return None
    
    ## <===================== HPTunning =====================>
    def _HyperParamSearch(self, X, y, search_space=None, search_method='grid', scoring='roc_auc', nFolds=5, printLog=True):
        ## count time
        beginTime = time.time()
        ## --------------------------------
        print(f"\tStart Hyper-Parameter Tunning ...")
        SearchResults = {'best_model': None, 'best_score':None, 'best_param':None}
        
        ##
        if search_method == 'grid':
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        elif search_method =='Bayes':
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)
        else:
            optimizer = GridSearchCV(estimator=self.model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=self._n_jobs)

        ## fit the Optimizer to the Data
        y_reshaped = y.to_numpy().reshape((len(y), ))
        optimizer.fit(X, y_reshaped)

        ## search results
        SearchResults['best_model'] = optimizer.best_estimator_
        SearchResults['best_score'] = optimizer.best_score_
        SearchResults['best_param'] = SearchResults['best_model'].get_params()
        self.HPT_Results[search_method] = SearchResults
        
        ##
        # self.model = optimizer.best_estimator_
        if SearchResults['best_param'] is not None:
            self.model.set_params(**SearchResults['best_param'])
        else:
            self.model = self.model

        if printLog:
            print(f"\tThis is the log info")
            print(f"\tThe best {scoring}: {SearchResults['best_score']}")
            print(f"\tThe optimized Params: {SearchResults['best_param']}")
            ## ----------------------------------------------------------------
            print(f"\tHyper-parameters Tunning costs time = {(time.time()-beginTime):.2f} s ................")
        return None
    
    ## <===================== tools =====================>    
    def __CalcScore_ROCAUC(self, y_prob, y_true, estTrsd=False):
        try:
            ## Assuming y_true are the true labels and y_prob are the predicted probabilities
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            aucs_score = auc(fpr, tpr)

            ## determine the best threshold
            if estTrsd:
                ## Calculate the distance to the top-left corner (0,1)
                distances = np.sqrt(fpr**2 + (1-tpr)**2)
                self.best_threshold = thresholds[distances.argmin()]
                print(f"\tThe best threshold is changged to {self.best_threshold}")
                # ## Calculate Youden's J statistic
                # youden_j = tpr - fpr
                # self.best_threshold = thresholds[youden_j.argmax()]
        except Exception as e:
            print(f'Warning! Cannot calculate ROC AUC, error msg: {e}')
            auc_score, fpr, tpr, thresholds = np.nan, np.nan, np.nan, np.nan
        results = {'auc_score': aucs_score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        return results

    def __CalcScore_ACC(self, y_pred, y_true):
        try:
            acc = accuracy_score(y_true, y_pred)
        except Exception as e:
            acc = np.nan
        return acc
    
    def __CalcScore_CM(self, y_pred, y_true):
        try:
            cm = confusion_matrix(Clas_test, Clas_test_pred)
        except Exception as e:
            cm = np.nan
        return cm
                
    def _CalcScores(self, y_pred, y_true, estTrsd=False, printLog=True):   
        dataDict_result = {}
        try:
            y_pred = y_pred.reshape((len(y_pred), ))
            y_true = y_true.reshape((len(y_true), ))
        except Exception as e:
            print(f"\tError! Cannot reformatting the y_pred and y_true when calculating the statistics")
        else:
            ## calculate the ROC auc
            dataDict_result['ROC_AUC'] = self.__CalcScore_ROCAUC(y_pred, y_true, estTrsd=estTrsd)
            
            ## calculate the accuracy
            y_pred_binary = np.where(y_pred >= self.best_threshold, 1, 0)
            dataDict_result['Accuracy'] = self.__CalcScore_ACC(y_pred=y_pred_binary, y_true=y_true)
            dataDict_result['ConfusionMatrics'] = self.__CalcScore_CM(y_pred=y_pred_binary, y_true=y_true)
             
            ## print out the results
            if printLog:
                print(f"\t\tData shape: y_pred {y_pred.shape}; y_true {y_true.shape}")
                print(f"\t\tAUROC: {dataDict_result['ROC_AUC']['auc_score']:.2f}")
                print(f"\t\tAccuracy: {dataDict_result['Accuracy']:.2f}")
                print(f"\t\tConfusionMatrics: {dataDict_result['ConfusionMatrics']}")
        return dataDict_result

    def _Plot_ROCAUC(self, ds_label):
        ## initiate the figure axes
        fig, ax = plt.subplots(figsize=(6, 6))
        ## generate plot
        try:
            dataDict_roc = self.performance[ds_label]['ROC_AUC']
            fpr, tpr, roc_auc = dataDict_roc['fpr'], dataDict_roc['tpr'], dataDict_roc['auc_score']
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=ds_label)
        except Exception as e:
            pass
        else:
            display.plot(ax=ax)
            ## set the figure config
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], 
                xlabel="False Positive Rate", ylabel="True Positive Rate", 
                title=f"ROC Curve ({ds_label})")
            ax.axis("square")
            ax.legend(loc="lower right")
            # plt.show()
        return fig
    
    def _Pred_Class(self, prob):
        if prob >= self.best_threshold:
            pred = 1 
        else:
            pred = 0
        return pred

    def ___futureFunctionsTBA():
        return None

#########################################################################################
################################# select_ML_methods #####################################
#########################################################################################

def select_ML_methods(modelType, ml_methed, rng=666666, knnk=3, n_jobs=-1):
    if modelType == 'regression':
        if ml_methed == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            sk_model = RandomForestRegressor(random_state=rng, oob_score=True, n_jobs=n_jobs)
            search_space = {
                'n_estimators': [50, 100, 250, 500],
                'max_depth': [2, 4, 6],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 5, 10, 25, 50],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 8, 10]}
        
        elif ml_methed == 'linear':
            from sklearn.linear_model import LinearRegression
            sk_model = LinearRegression(n_jobs=n_jobs)
            search_space = None
            # from sklearn.linear_model import Lasso
            # sk_model = Lasso(alpha=0.1)
            # search_space = {'alpha': [0, 0.1, 0.25, 0.5, 0.8]}
        
        elif ml_methed == 'SVM':
            from sklearn.svm import SVR
            sk_model = SVR(kernel="rbf", gamma=0.1)
            search_space = {
                'kernel': ['poly', 'rbf', 'sigmoid'], 
                'gamma': ['scale', 'auto'], 
                'C': [0.1, 1, 10, 100]}
        
        elif ml_methed == 'MLP':
            from sklearn.neural_network import MLPRegressor
            sk_model = MLPRegressor(random_state=rng, max_iter=500, early_stopping=True)
            search_space = {
                'hidden_layer_sizes': [(128,), (128, 128), (128, 128, 128)], 
                'activation': ['logistic', 'tanh', 'relu'], 
                'solver': ['sgd', 'adam'],
                'alpha': [0.1, 0.01, 0.001, 0.0001]}
        
        elif ml_methed == 'KNN':
            from sklearn.neighbors import KNeighborsRegressor
            sk_model = KNeighborsRegressor(n_neighbors=knnk, n_jobs=n_jobs)
            search_space = {'n_neighbors': [1, 3, 5, 10]}
        
        else:
            print(f"Error! no proper ML methods were selected, using Linear method instead")
            from sklearn.linear_model import Lasso
            sk_model = Lasso(alpha=0.1)
            search_space = {'alpha': [0, 0.1, 0.25, 0.5, 0.8]}

    elif modelType == 'classification':
        if ml_methed == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            sk_model = RandomForestClassifier(random_state=rng, class_weight='balanced_subsample', oob_score=True, n_jobs=n_jobs)
            search_space = {
                'n_estimators': [50, 100, 250, 500],
                'max_depth': [2, 4, 6],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 5, 10, 25, 50],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 8, 10]}

        elif ml_methed == 'linear':
            from sklearn.linear_model import LogisticRegression
            sk_model = LogisticRegression(random_state=rng, n_jobs=n_jobs)
            search_space = None

        elif ml_methed == 'SVM':
            from sklearn.svm import SVC
            sk_model = SVC(kernel="rbf", gamma=0.1, random_state=rng, probability=True)
            search_space = {
                'kernel': ['poly', 'rbf', 'sigmoid'], 
                'gamma': ['scale', 'auto'], 
                'C': [0.1, 1, 10, 100]}

        elif ml_methed == 'MLP':
            from sklearn.neural_network import MLPClassifier
            sk_model = MLPClassifier(random_state=rng, max_iter=500, early_stopping=True)
            search_space = {
                'hidden_layer_sizes': [(128,), (128, 128), (128, 128, 128)], 
                'activation': ['logistic', 'tanh', 'relu'], 
                'solver': ['sgd', 'adam'],
                'alpha': [0.1, 0.01, 0.001, 0.0001]}
        
        elif ml_methed == 'XGBoost':
            from sklearn.ensemble import GradientBoostingClassifier
            sk_model = GradientBoostingClassifier(n_estimators=100, random_state=rng)
            search_space = {
                'n_estimators': [50, 100, 250, 500],
                'loss': ["log_loss", "exponential"],
                'max_depth': [1, 3, 5],
                'learning_rate': [0.01, 0.1, 1],
                'min_samples_leaf': [1, 5, 10, 25, 50],
                'min_samples_split': [2, 5, 8, 10],
                'max_features': ['sqrt', 'log2', None]}
        
        elif ml_methed == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            sk_model = KNeighborsClassifier(n_neighbors=knnk, n_jobs=n_jobs)
            search_space = {'n_neighbors': [1, 3, 5, 10]}

    else:
        print(f"\tError! ML model type should be one of <regression> or <classification>" )
    return sk_model, search_space