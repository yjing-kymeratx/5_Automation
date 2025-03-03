'''
To be added
'''
####################################################################
####################### descriptor filters #########################
####################################################################
## ------------ remove descriptors with too many missing data ------------
def missingValueFilter(desc_all, json_file_imput_param, nan_cutoff=0.2):
    import pandas as pd
    nan_ratio_dict, desc_sele = {}, []
    ## load imputation param file
    import os, json
    if os.path.exists(json_file_imput_param):
        with open(json_file_imput_param, 'r') as ifh:
            dict_imput_param = json.load(ifh)
        ##
        for desc in dict_imput_param:
            if desc in desc_all:
                count_nan = dict_imput_param[desc]['count_nan']
                count_all = dict_imput_param[desc]['count_all']
                nan_ratio = count_nan/count_all
                if desc not in nan_ratio_dict:
                    nan_ratio_dict[desc] = {}
                    nan_ratio_dict[desc]['Descriptor'] = desc
                    nan_ratio_dict[desc]['nan_ratio'] = nan_ratio
                if nan_ratio <= nan_cutoff:
                    nan_ratio_dict[desc]['Select'] = 'Yes'
                    desc_sele.append(desc)
                else:
                    nan_ratio_dict[desc]['Select'] = 'No'
        ## print results
        print(f"\t\tIn total <{len(desc_all)}> desc, there are <{len(desc_sele)}> selected, cutoff is {nan_cutoff}.\n")
    else:
        print(f"Error! The imputation param file {json_file_imput_param} does not exist\n")

    nan_ratio_table = pd.DataFrame.from_dict(nan_ratio_dict).T
    return nan_ratio_table, desc_sele

## ------------ Variance-based Filter ------------
def VarianceFilter(X, threshold=0):
    import pandas as pd
    ## Removing features with no variance
    from sklearn.feature_selection import VarianceThreshold
    variance_filter = VarianceThreshold(threshold=threshold)
    variance_filter.fit_transform(X)

    ## generate variance table
    desc_list = list(X.columns)
    variance_list = variance_filter.variances_.tolist()
    variance_dict, desc_sele = {}, []
    for i in range(len(desc_list)):
        variance_dict[i] = {}
        variance_dict[i]['Descriptor'] = desc_list[i]
        variance_dict[i]['Variance'] = variance_list[i]
        if variance_list[i] > threshold:
            variance_dict[i]['Select'] = 'Yes'
            desc_sele.append(desc_list[i])
        else:
            variance_dict[i]['Select'] = 'No'
    variance_table = pd.DataFrame.from_dict(variance_dict).T
    ## print results
    print(f"\t\tVarianceFilter: In total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected, cutoff is {threshold}.\n")
    return variance_table, desc_sele

## ------------ L2-based Filter ------------
def L2_based_selection(X, y, modelType='regression', penalty_param=0.01):
    import pandas as pd
    ## define estimator model
    if modelType == 'regression':
        from sklearn import linear_model
        print(f"\t\tThe model type is {modelType} so select Lasso model.\n")
        estimator = linear_model.Lasso(alpha=penalty_param, max_iter=5000, random_state=666666)
    elif modelType == 'classification':
        from sklearn.svm import LinearSVC
        print(f"\t\tThe model type is {modelType} so select LinearSVC model.\n")
        estimator = LinearSVC(penalty="l2", loss="squared_hinge", dual=True)
    else:
        print(f"\tError! Your model type <{modelType}> is not in [regression, classification]\n")
    
    ## fit estimator model
    y_np = y.to_numpy().reshape((len(y),))
    estimator.fit(X, y_np)
    scores = estimator.coef_

    ## selector model
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(estimator=estimator, prefit=True)
    select_feature_mask = selector.get_support()

    ## result table
    desc_list = list(X.columns)
    coef_dict, desc_sele = {}, []
    if len(scores) == len(desc_list):
        for i in range(len(desc_list)):
            coef_dict[i] = {}
            coef_dict[i]['Descriptor'] = desc_list[i]
            coef_dict[i]['l2_coef'] = scores[i]
            if select_feature_mask[i]:
                coef_dict[i]['Select'] = 'Yes'
                desc_sele.append(desc_list[i])
            else:
                coef_dict[i]['Select'] = 'No'
    else:
        print(f"Error! The desc (N={len(desc_list)}) does not match the coef scores (N={len(scores)})\n")

    ## print results
    print(f"\t\tL2 filter: In total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected.\n")
    coef_table = pd.DataFrame.from_dict(coef_dict).T
    return coef_table, desc_sele

## ------------ RF-based Filter ------------
def RF_based_selection(X, y, modelType='regression', penalty_param=0.01):
    import pandas as pd
    ## define estimator model
    if modelType == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(random_state=666666)
    elif modelType == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier()
    
    ## fit estimator model
    y_np = y.to_numpy().reshape((len(y),))
    estimator.fit(X, y_np)
    scores = estimator.feature_importances_

    ## selector model
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(estimator=estimator, prefit=True)
    select_feature_mask = selector.get_support()

    ## result table
    desc_list = list(X.columns)
    FI_dict, desc_sele = {}, []
    if len(scores) == len(desc_list):
        for i in range(len(desc_list)):
            FI_dict[i] = {}
            FI_dict[i]['Descriptor'] = desc_list[i]
            FI_dict[i]['feature_importance'] = scores[i]
            if select_feature_mask[i]:
                FI_dict[i]['Select'] = 'Yes'
                desc_sele.append(desc_list[i])
            else:
                FI_dict[i]['Select'] = 'No'
    else:
        print(f"Error! The desc (N={len(desc_list)}) does not match the feature importance (N={len(scores)})\n")

    ## print results
    print(f"\t\tRF filter: In total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected.\n")
    FI_table = pd.DataFrame.from_dict(FI_dict).T
    return FI_table, desc_sele

####################################################################
########################## Tools ###################################
####################################################################
## get the args
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)

    # parser.add_argument('-i', '--input', action="store", default='./results', help='The input folder of all desciptor files')
    parser.add_argument('-x', '--desc', action="store", default="./Results/descriptors_prep_merged.csv", help='The input file of the desciptor file')
    parser.add_argument('-y', '--expt', action="store", default="./Results/outcome_expt.csv", help='The input file of the experiment outcome file')
    parser.add_argument('-s', '--split', action="store", default="./Results/data_split.csv", help='The input file of the train/val/test split file')    
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colAssay', action="store", default=None, help='The column name of the experiment outcome')
    parser.add_argument('--cols', action="store", default='Split', help='The column name of the split')

    parser.add_argument('--modelType', action="store", default="regression", help='ML model type, either <regression> or <classification>')
    parser.add_argument('--MissingValueFilter', action="store", default="True", help='remove the descriptor with a lot of missing values')
    parser.add_argument('--impuParamJson', action="store", default="./Results/feature_imputation_params.json", help='The json file of descriptor imputation parameters')

    parser.add_argument('--VarianceFilter', action="store", default="True", help='remove the descriptor with low variance')
    parser.add_argument('--L2Filter', action="store", default="True", help='feature selection using linear Lasso method')
    parser.add_argument('--FIFilter', action="store", default="True", help='feature selection using RF feature importance')

    parser.add_argument('-o', '--output', action="store", default="./Results/data_input_4_ModelBuilding.csv", help='save the selected desc csv file')

    args = parser.parse_args()
    return args

##
def run_script(input_X, input_y, input_split, sep=',', colName_mid='Compound Name', colName_split='Split', colName_expt=None, 
               modelType="regression", doMissingValueFilter=True, doVarianceFilter=True, doL2Filter=True, doFeatureImportanceFilter=True, 
               json_file_imput_param="./Results/feature_imputation_params.json", 
               filePathOut="./Results/data_input_4_ModelBuilding.csv"):

    print(f">>>>Selecting important Descriptors ...")
    ## ------------ load data ------------
    ## load all descriptor tables and merge together
    import pandas as pd
    dataTable_s = pd.read_csv(input_split, sep=sep, usecols=[colName_mid, colName_split])
    dataTable_y = pd.read_csv(input_y, sep=sep, usecols=[colName_mid, colName_expt])
    dataTable_X = pd.read_csv(input_X, sep=sep)
    print(f"\tLoading split DataFrame: <{dataTable_s.shape}>, y DataFrame: <{dataTable_y.shape}>, X DataFrame: <{dataTable_X.shape}>\n")

    dataTable_merged_all = pd.merge(left=dataTable_s, right=dataTable_y, on=colName_mid, how='inner')
    dataTable_merged_all = dataTable_merged_all.merge(right=dataTable_X, on=colName_mid, how='inner')
    print(f"\tThe merged table has shape {dataTable_merged_all.shape}\n")

    ## select the training&validation data
    dataTable_merged = dataTable_merged_all[~(dataTable_merged_all[colName_split].isin(['Test']))]
    # dataTable_merged = dataTable_merged.drop(columns=[colName_split])
    print(f"\tThe training/validation table has shape {dataTable_merged.shape}\n")

    ## get descriptors (X) and outcome (y)
    colNames_X = [desc for desc in dataTable_X.columns if desc != colName_mid]
    X, y = dataTable_merged[colNames_X], dataTable_merged[colName_expt]
    print(f"\tX has shape {X.shape}, y has shape {y.shape}\n")

    ## ------------ filters ------------
    score_table_dict = {}
    ## remove descriptors with too many missing data
    if doMissingValueFilter:
        print(f"\tremove descriptors with too many missing data\n")
        score_table_dict['MissingValueFilter'], desc_sele = missingValueFilter(list(X.columns), json_file_imput_param, nan_cutoff=0.2)
        X = X[desc_sele]
    else:
        score_table_dict['MissingValueFilter'], desc_sele = missingValueFilter(list(X.columns), json_file_imput_param, nan_cutoff=0.2)
        X = X[desc_sele]

    ## Variance-based Filter
    if doVarianceFilter:
        print(f"\tremove descriptors with too low variance\n")
        score_table_dict['VarianceFilter'], desc_sele = VarianceFilter(X=X, threshold=0.001)
        X = X[desc_sele]

    ## L2-based Filter
    if doL2Filter:
        print(f"\tremove descriptors with L2 regulization\n")
        score_table_dict['L2Filter'], desc_sele = L2_based_selection(X=X, y=y, modelType=modelType)
        X = X[desc_sele]

    ## RF-based Filter
    if doFeatureImportanceFilter:
        print(f"\tremove descriptors with RF feature importance\n")
        score_table_dict['FIFilter'], desc_sele = RF_based_selection(X=X, y=y, modelType=modelType)
        X = X[desc_sele]

    ## ------------------ clean the dataset ------------------
    dataTable_merged_clean = dataTable_merged_all[[colName_mid, colName_split, colName_expt] + desc_sele]
    print(f"\tThere are total <{len(desc_sele)}> descriptors selected for ML modeling")
    dataTable_merged_clean.to_csv(filePathOut, index=False)
    print(f"\tThe cleaned data table for ML model building is saved to <{filePathOut}>")

    ## ------------------ merge all stats tables and save to csv ------------------
    score_table_merged = pd.DataFrame(columns=['Descriptor'])
    for filterType in score_table_dict:
        this_Table = score_table_dict[filterType]
        
        this_Table = this_Table.rename(columns={'Select': f'select_{filterType}'})
        score_table_merged = score_table_merged.merge(right=this_Table, on='Descriptor', how='outer')
    score_table_merged['Select_final'] = score_table_merged['Descriptor'].apply(lambda x: 'Yes' if x in desc_sele else 'No')

    ## output folder
    import os
    folderPathOut = os.path.dirname(filePathOut)    ## './results'
    os.makedirs(folderPathOut, exist_ok=True)       
    fileNameOut_FSscore = f"{folderPathOut}/feature_scoring.csv"
    score_table_merged.to_csv(fileNameOut_FSscore, index=False)
    print(f"\tThe merged table contains all feature selection scores is saved to <{fileNameOut_FSscore}>")

    return filePathOut

####################################################################
######################### main function ############################
####################################################################
def main():
    ## ------------ load args ------------
    args = Args_Prepation(parser_desc='Feature selection')
    
    input_X = args.desc
    input_y = args.expt
    input_split = args.split
    sep = args.delimiter    # ',' 
    colName_mid = args.colId    # 'Compound Name'
    colName_split = args.cols
    colName_expt = args.colAssay

    modelType = args.modelType    # 'regression', 'classification'
    doMissingValueFilter = True if args.MissingValueFilter in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False
    json_file_imput_param = args.impuParamJson
    doVarianceFilter = True if args.VarianceFilter in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False
    doL2Filter = True if args.L2Filter in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False
    doFeatureImportanceFilter = True if args.FIFilter in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False
    filePathOut = args.output

    ## 
    run_script(input_X, input_y, input_split, sep, colName_mid, colName_split, colName_expt, modelType,
               doMissingValueFilter, doVarianceFilter, doL2Filter, doFeatureImportanceFilter,
               json_file_imput_param, filePathOut)

if __name__ == '__main__':
    main()