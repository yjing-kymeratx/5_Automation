'''
-i "./results/data_input_4_ModelBuilding.csv"
-d ','
--colId 'Compound Name'
--cols 'Split'
--coly 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'

--modelType "regression"
--linear "True"
--rf "True"
--svm "True"
--mlp "True"
--knn "True"

--njobs -1
--rng 666666
--logy "True"
--doHPT "True"

-o "./results"

--calcParamJson f"{folderPathOut}/calculator_param.json"
--normParamJson f"{folderPathOut}/feature_normalization_params.json"
--impuParamJson f"{folderPathOut}/feature_imputation_params.json"

'''

####################################################################
######################## model building ############################
####################################################################

## <===================== model initiate =====================>
def step_1_model_init(ml_methed, n_jobs=-1, rng=666666):
    ml_methed = ml_methed.lower()
    ## -------------------- random forest --------------------
    if ml_methed in ['rf', 'random forest', 'randomforest']:
        from sklearn.ensemble import RandomForestRegressor
        sk_model = RandomForestRegressor(random_state=rng, oob_score=True, n_jobs=n_jobs)
        search_space = {'n_estimators': [50, 200, 500], 'max_depth': [2, 4, 6], 'max_features': ['sqrt', 'log2'], 'min_samples_leaf': [5, 10, 25, 50], 'min_samples_split': [2, 5, 8, 10]}

    ## -------------------- SVM --------------------
    elif ml_methed in ['svm', 'support vector machine', 'supportvectormachine']:
        from sklearn.svm import SVR
        sk_model = SVR(kernel="rbf", gamma=0.1)
        search_space = {'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'C': [0.1, 1, 10, 100]}

    ## -------------------- MLP --------------------
    elif ml_methed in ['mlp', 'ann']:
        from sklearn.neural_network import MLPRegressor
        sk_model = MLPRegressor(random_state=rng, max_iter=100, early_stopping=True)
        search_space = {'hidden_layer_sizes': [(128,), (128, 128), (128, 128, 128)], 'activation': ['logistic', 'tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.1, 0.01, 0.001, 0.0001]}

    ## -------------------- KNN --------------------
    elif ml_methed in ['knn', 'k-nn', 'nearest neighbor', 'nearestneighbor']:
        from sklearn.neighbors import KNeighborsRegressor
        sk_model = KNeighborsRegressor(n_neighbors=3, n_jobs=n_jobs)
        search_space = {'n_neighbors': [1, 3, 5, 10]}

    ## -------------------- Linear --------------------
    else:
        if ml_methed != 'linear':
            print(f"Error! no proper ML methods were selected, using Linear method instead")
        from sklearn.linear_model import LinearRegression
        sk_model = LinearRegression(n_jobs=n_jobs)
        search_space = None

    return sk_model, search_space

## <===================== model training =====================>
def _HyperParamSearch(sk_model, X, y, search_space=None, search_method='grid', scoring='neg_mean_absolute_error', nFolds=5, n_jobs=-1):
    print(f"\t\tStart Hyper-Parameter Tunning ...")
    SearchResults = {'best_model': None, 'best_score':None, 'best_param':None}

    # if search_method == 'grid':
    # Mute warining
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from sklearn.model_selection import GridSearchCV
    optimizer = GridSearchCV(estimator=sk_model, param_grid=search_space, scoring=scoring, cv=nFolds, n_jobs=n_jobs)
    optimizer.fit(X, y)
    ## search results
    SearchResults['optimizer'] = optimizer    ## optimizer.best_estimator_, optimizer.best_score_
    SearchResults['best_param'] = optimizer.best_estimator_.get_params()

    ## export
    print(f"\t\tThe best {scoring}: {SearchResults['best_score']}\n\t\tThe best hp-params: {SearchResults['best_param']}")
    print(f"\t\tComplete Hyper-Parameter Tunning ...")
    return SearchResults

##
def step_2_model_training(sk_model, X, y, logy=False, doHPT=False, search_space=None, scoring='neg_mean_absolute_error', n_jobs=-1):
    import time   
    beginTime = time.time()        
    ## ----------------------------------------------------------------
    import numpy as np
    # X = X.to_numpy()
    y = y.to_numpy().reshape((len(y), ))
    y = np.log10(y) if logy else y

    ## ----- hyper parameter search ----------------
    if doHPT and search_space is not None:
        HPSearchResults = _HyperParamSearch(sk_model, X, y, search_space, search_method='grid', scoring=scoring, nFolds=5, n_jobs=n_jobs)
        sk_model = sk_model.set_params(**HPSearchResults['best_param'])    #optimizer.best_estimator_

    ## ----- fit the model -----
    sk_model.fit(X, y)

    ## ----------------------------------------------------------------        
    print(f"\t\tThe model training costs time = {(time.time()-beginTime):.2f} s ................")
    return sk_model

## <===================== model predict =====================>
def step_3_make_prediction(sk_model, X, logy=False):
    y_pred = sk_model.predict(X)
    y_pred = 10**y_pred if logy else y_pred
    return y_pred

## <===================== model evaluate =====================>
def step_4_make_comparison(y_pred, y_true, dsLabel='dataset'):
    dataDict_evaluation = {}
    try:
        y_pred = y_pred.reshape((len(y_pred), ))
        y_true = y_true.reshape((len(y_true), ))

        from sklearn.metrics import mean_absolute_error
        from numpy import corrcoef
        from scipy.stats import spearmanr, kendalltau
        dataDict_evaluation[f'DataSet'] = dsLabel
        dataDict_evaluation[f'MAE'] = float(mean_absolute_error(y_pred, y_true))    ## mean absolute error
        dataDict_evaluation[f'Pearson_R2'] = float(corrcoef(y_pred, y_true)[0, 1])**2    ## PearsonCorrelationCoefficient
        dataDict_evaluation[f'Spearman_R2'] = float(spearmanr(y_pred, y_true)[0])**2    ## rank-order correlation
        dataDict_evaluation[f'KendallTau_R2'] = float(kendalltau(y_pred, y_true)[0])**2        ## Kendall's tau
    except Exception as e:
        print(f"\t\tCannot compare y_pred & y_true: {e}")
    else:          
        ## print out the results
        mae = round(dataDict_evaluation[f'MAE'], 2)
        pr2 = round(dataDict_evaluation[f'Pearson_R2'], 2)
        sr2 = round(dataDict_evaluation[f'Spearman_R2'], 2)
        print(f"\t\t{dsLabel}=> MAE: {mae}; Pearson-R2: {pr2}; Spearman-R2: {sr2}")
        # print(f"\t\tKendall-R2: {dataDict_evaluation['KendallTau_R2']:.2f}")
    return dataDict_evaluation

## <===================== resultl plot =====================>
def step_5_plot_pred_vs_expt(dataTable, colName_x='Prediction', colName_y='Experiment', colName_color='Split', take_log=True, diagonal=True, sideHist=True, figTitle=None, outputFolder='./Plot'):
    ## --------- Start with a square Figure ---------
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    x, y = dataTable[colName_x], dataTable[colName_y]
    if take_log:
        import numpy as np
        x, y = np.log10(x), np.log10(y)
        label_x = f"Prediction(log)" if take_log else 'Prediction'    # f"{colName_x}(log)" if take_log else colName_x
        label_y = f"Experiment(log)" if take_log else 'Experiment'    # f"{colName_y}(log)" if take_log else colName_y

    ## --------- add histgram along each side of the axis ---------
    if sideHist:
        gs = fig.add_gridspec(2, 2,  width_ratios=(6, 1), height_ratios=(1, 6), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ## add hist
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        bins = 20        
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')
            
        ax_histx.tick_params(axis="x", labelbottom=False)    # no x labels
        ax_histy.tick_params(axis="y", labelleft=False)    # no y labels

        ax_histx.tick_params(axis='both', which='major', labelsize=16)
        ax_histy.tick_params(axis='both', which='major', labelsize=16)
    else:
        ax = fig.add_subplot()
        
    ## --------- add scatter plot ---------
    if colName_color is None:
        ax.scatter(x, y, s=40, alpha=0.5, cmap='Spectral', marker='o')
    else:
        for i in sorted(dataTable[colName_color].unique()):
            idx = dataTable[dataTable[colName_color]==i].index.to_list()
            ax.scatter(x.loc[idx], y.loc[idx], s=40, alpha=0.5, cmap='Spectral', marker='o', label=i)
        ax.legend(loc="best", title=colName_color)    #, bbox_to_anchor=(1.35, 0.5)
        # ax.legend(loc="upper left", title=colName_color)    #, bbox_to_anchor=(1.35, 0.5)

        
    ## --------- config the figure params ---------
    ax.set_xlabel(label_x, fontsize=16)
    ax.set_ylabel(label_y, fontsize=16)

    ## determine axis limits  
    ax_max, ax_min = max(np.max(x), np.max(y)), min(np.min(x), np.min(y))
    ax_addon = (ax_max - ax_min)/10
    ax_max = ax_max + ax_addon
    ax_min = ax_min - ax_addon
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(alpha=0.75)
    if diagonal: 
        diagonalLine = ax.plot([ax_min, ax_max], [ax_min, ax_max], c='lightgray', linestyle='-')        

    figTitle = f"Pred vs Expt" if figTitle is None else figTitle
    fig.suptitle(figTitle, fontsize=24)

    ## --------- Save the figure ---------
    import os
    os.makedirs(outputFolder, exist_ok=True)
    figPath = f"{outputFolder}/{figTitle}.png"
    plt.savefig(figPath)
    return fig

## <===================== others =====================>


####################################################################
########################## Tools ###################################
####################################################################
## get the args
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)

    # parser.add_argument('-i', '--input', action="store", default='./results', help='The input folder of all desciptor files')
    parser.add_argument('-i', '--input', action="store", default="./Results/data_input_4_ModelBuilding.csv", help='The input file for model building')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--cols', action="store", default='Split', help='The column name of the split')
    parser.add_argument('--coly', action="store", default=None, help='The column name of the experiment outcome')
    parser.add_argument('--modelType', action="store", default="regression", help='ML model type, either <regression> or <classification>')
    parser.add_argument('--linear', action="store", default="True", help='linear methods')
    parser.add_argument('--rf', action="store", default="True", help='rf methods')
    parser.add_argument('--svm', action="store", default="True", help='svm methods')
    parser.add_argument('--mlp', action="store", default="True", help='mlp methods')
    parser.add_argument('--knn', action="store", default="True", help='knn methods')

    parser.add_argument('--rng', action="store", default="666666", help='random seed for the random process')
    parser.add_argument('--njobs', action="store", default="-1", help='remove the descriptor with a lot of missing values')

    parser.add_argument('--logy', action="store", default="True", help='take the log of y')
    parser.add_argument('--doHPT', action="store", default="True", help='do hyperparameter tunning')

    parser.add_argument('--calcParamJson', action="store", default="./Results/feature_calculator_param.json", help='The json file of descriptor calculater parameters')
    parser.add_argument('--normParamJson', action="store", default="./Results/feature_normalization_params.json", help='The json file of descriptor normalization parameters')
    parser.add_argument('--impuParamJson', action="store", default="./Results/feature_imputation_params.json", help='The json file of descriptor imputation parameters')

    parser.add_argument('-o', '--output', action="store", default="./Results/performance_results.csv", help='save the modeling performance file')

    args = parser.parse_args()
    return args

####################################################################
######################### main function ############################
####################################################################
def main():
    print(f">>>>Building ML models ...")
    # Mute warining
    import warnings
    warnings.filterwarnings("ignore")
    
    ## ------------ load args ------------
    args = Args_Prepation(parser_desc='ML model building')

    if True:
        fileNameIn = args.input
        sep = args.delimiter
        colName_mid = args.colId
        colName_split = args.cols
        colName_y = args.coly

        modelType = args.modelType
        ml_method_list = []
        if args.linear == 'True':
            ml_method_list.append('linear')
        if args.rf == 'True':
            ml_method_list.append('rf')
        if args.svm == 'True':
            ml_method_list.append('svm')
        if args.mlp == 'True':
            ml_method_list.append('mlp')
        if args.knn == 'True':
            ml_method_list.append('knn')

        rng = int(args.rng)
        n_jobs = int(args.njobs)
        logy = True if args.logy == 'True' else False
        doHPT = True if args.doHPT == 'True' else False

        ## output folder
        import os
        filePathOut = args.output 
        folderPathOut = os.path.dirname(filePathOut)    ## './results'
        os.makedirs(folderPathOut, exist_ok=True)   

        ## model folder
        folderPathOut_model = f"{folderPathOut}/Models"
        os.makedirs(folderPathOut_model, exist_ok=True)

        ## plot folder
        folderPathOut_plot = f"{folderPathOut}/Plots"
        os.makedirs(folderPathOut_plot, exist_ok=True)

        ## ------------ load params for desc calc ------------
        import json
        with open(args.calcParamJson, 'r') as ifh_dc:
            desc_calc_param = json.load(ifh_dc)
        with open(args.normParamJson, 'r') as ifh_dn:
            desc_norm_param = json.load(ifh_dn)
        with open(args.impuParamJson, 'r') as ifh_di:
            desc_impu_param = json.load(ifh_di)

    ## ------------------------ load data ------------------------
    import pandas as pd
    dataTable_raw = pd.read_csv(fileNameIn, sep=sep)
    colName_X = [col for col in dataTable_raw.columns if col not in [colName_mid, colName_split, colName_y]]

    if True:
        ## training
        dataTable_train = dataTable_raw[dataTable_raw[colName_split]=='Training']
        X_train, y_train = dataTable_train[colName_X], dataTable_train[colName_y]
        print(f"\tTraining_X: {X_train.shape}; Training_y: {y_train.shape}")

        ## validation
        dataTable_val = dataTable_raw[dataTable_raw[colName_split]=='Validation']
        X_val, y_val = dataTable_val[colName_X], dataTable_val[colName_y]
        print(f"\tValidation_X: {X_val.shape}; Validation_y: {y_val.shape}")

        ## test
        dataTable_test = dataTable_raw[dataTable_raw[colName_split]=='Test']
        X_test, y_test = dataTable_test[colName_X], dataTable_test[colName_y]
        print(f"\tTest_X: {X_test.shape}; Test_y: {y_test.shape}")

    ## ------------------------ models ------------------------ 
    model_dict_all = {}
    model_dict_performance = {}
    scoring = 'neg_mean_absolute_error'
    ##
    for ml_methed in ml_method_list:
        model_dict = {'data': {'training': dataTable_train[colName_mid], 'validation': dataTable_val[colName_mid], 'test': dataTable_test[colName_mid]},
                      'desc': colName_X, 'exptOutcome': colName_y,
                      'param': {'calculator': desc_calc_param, 'normalization': desc_norm_param, 'imputation': desc_impu_param},
                      'config': {'rng': rng, 'n_jobs':n_jobs, 'logy': logy, 'doHPT': doHPT}, 
                      'model': None, 'results': None, 'performance_dict': {}, 'plot': None}


        print(f"\t>>Now training the <{ml_methed}> model")
        ## ------------ training ------------
        sk_model, search_space = step_1_model_init(ml_methed, n_jobs=n_jobs, rng=rng)
        sk_model = step_2_model_training(sk_model, X_train, y_train, logy=logy, doHPT=doHPT, search_space=search_space, scoring=scoring, n_jobs=n_jobs) 
        model_dict['model'] = sk_model

        ## ------------ prediction ------------
        col_pred = f"Prediction_{ml_methed}_{colName_y}"
        dataTable_train[col_pred] = step_3_make_prediction(sk_model, X_train, logy=logy)
        dataTable_val[col_pred] = step_3_make_prediction(sk_model, X_val, logy=logy)
        dataTable_test[col_pred] = step_3_make_prediction(sk_model, X_test, logy=logy)

        ## ------------ evaluation stats ------------
        model_dict['performance_dict']['Training'] = step_4_make_comparison(dataTable_train[col_pred].to_numpy(), dataTable_train[colName_y].to_numpy(), dsLabel='Training')
        model_dict['performance_dict']['Validation'] = step_4_make_comparison(dataTable_val[col_pred].to_numpy(), dataTable_val[colName_y].to_numpy(), dsLabel='Validation')
        model_dict['performance_dict']['Test'] = step_4_make_comparison(dataTable_test[col_pred].to_numpy(), dataTable_test[colName_y].to_numpy(), dsLabel='Test')
        model_dict['performance'] = pd.DataFrame.from_dict(model_dict['performance_dict']).T
        
        ## ------------ plot pred vs expt ------------
        model_dict['plot'] = step_5_plot_pred_vs_expt(dataTable=pd.concat([dataTable_train, dataTable_val, dataTable_test]), 
                                                      colName_x=col_pred, colName_y=colName_y, colName_color=colName_split,
                                                      take_log=True, diagonal=True, sideHist=True, outputFolder=folderPathOut_plot, 
                                                      figTitle=f"Predictedn_VS_Experimental_{ml_methed}")
        ##        
        model_dict_all[ml_methed] = model_dict

    ## ------------ merge/concact & save prediction data ------------
    dataTable_pred_all = pd.concat([dataTable_train, dataTable_val, dataTable_test]).sort_index(ascending=True)
    ## save
    fileNameOut_pred = f"{folderPathOut}/prediction_results.csv"
    dataTable_pred_all.to_csv(fileNameOut_pred, index=False)
    print(f"\tThe prediction data are saved to <{fileNameOut_pred}>")

    ## ------------ merge/concact & save performance data ------------
    dataDict_perform_all = {}
    for ml_methed in model_dict_all:
        dataDict_perform_all[ml_methed] = {}
        dataDict_perform_all[ml_methed]['ML_Algorithm'] = ml_methed

        perform_dict = model_dict_all[ml_methed]['performance_dict']
        for dsLable in ['Training', 'Validation', 'Test']:
            for mtx in perform_dict[dsLable]:
                if mtx not in ['DataSet']:
                    dataDict_perform_all[ml_methed][f"{dsLable}_{mtx}"] = perform_dict[dsLable][mtx]
    dataTable_perform_all = pd.DataFrame.from_dict(dataDict_perform_all).T
    ## save
    fileNameOut_perf = filePathOut    # f"{folderPathOut}/performance_results.csv"
    dataTable_perform_all.to_csv(fileNameOut_perf, index=False)
    print(f"\tThe performance/evaluation data are saved to <{fileNameOut_perf}>")
    
    ## ------------ find best model ------------
    # select_matrics, select_standar = 'Validation_MAE', 'min'
    # select_matrics, select_standar = 'Validation_Pearson_R2', 'max'
    select_matrics, select_standar = 'Validation_Spearman_R2', 'max'
    if select_standar in ['min']:
        ml_sele = dataTable_perform_all.loc[dataTable_perform_all[select_matrics].idxmin()]['ML_Algorithm']
    elif select_standar in ['max']:
        ml_sele = dataTable_perform_all.loc[dataTable_perform_all[select_matrics].idxmax()]['ML_Algorithm']
    else:
        print(f"\tThe select_standar should be selected from [max, min]\n")
        ml_sele = 'To be determined'
    model_dict_all['Best_AutoML'] = model_dict_all[ml_sele] if ml_sele in model_dict_all else {}
    print(f"\tThe BEST model is {ml_methed} model.\n")

## ------------ save & export model ------------
    import pickle
    for ml_methed in model_dict_all:
        if model_dict_all[ml_methed] is not None:
            fileNameOut_model = f"{folderPathOut_model}/{ml_methed}_models.pickle"
            with open(fileNameOut_model, 'wb') as ofh_models:
                pickle.dump(model_dict, ofh_models)
                print(f"\tThe {ml_methed} model is saved to <{fileNameOut_model}>\n")

if __name__ == '__main__':
    main()