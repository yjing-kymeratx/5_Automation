'''
--input     './results'
--delimiter ','
--colId     'Compound Name'

--desc_custom       True
--desc_rdkit        True
--desc_fps      True
--desc_cx       True

--modelType     'regression'    # 'classification'

--MissingValueFilter        True
--VarianceFilter        True
--L2Filter      True
--FeatureImportanceFilter       True
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
        print(f"\t\tIn total <{len(nan_ratio_dict)}> desc, there are <{len(desc_sele)}> selected, cutoff is {nan_cutoff}.")
    else:
        print(f"Error! The imputation param file {json_file_imput_param} does not exist")

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
    print(f"\t\tIn total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected, cutoff is {threshold}.")
    return variance_table, desc_sele

## ------------ L2-based Filter ------------
def L2_based_selection(X, y, model_type='regression', penalty_param=0.01):
    import pandas as pd
    ## define estimator model
    if model_type == 'regression':
        from sklearn import linear_model
        estimator = linear_model.Lasso(alpha=penalty_param)
    elif model_type == 'classification':
        from sklearn.svm import LinearSVC
        estimator = LinearSVC(penalty="l2", loss="squared_hinge", dual=True)
    
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
        print(f"Error! The desc (N={len(desc_list)}) does not match the coef scores (N={len(scores)})")

    ## print results
    print(f"\t\tIn total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected.")
    coef_table = pd.DataFrame.from_dict(coef_dict).T
    return coef_table, desc_sele

## ------------ RF-based Filter ------------
def RF_based_selection(X, y, model_type='regression', penalty_param=0.01):
    import pandas as pd
    ## define estimator model
    if model_type == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor()
    elif model_type == 'classification':
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
        print(f"Error! The desc (N={len(desc_list)}) does not match the feature importance (N={len(scores)})")

    ## print results
    print(f"\t\tIn total <{len(desc_list)}> desc, there are <{len(desc_sele)}> selected.")
    FI_table = pd.DataFrame.from_dict(FI_dict).T
    return FI_table, desc_sele

####################################################################
########################## Tools ###################################
####################################################################
## get the args
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument('-i', '--input', action="store", default='./results', help='The input folder of all desciptor files')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')

    parser.add_argument('--desc_custom', action="store", default="True", help='use the custom descriptors')
    parser.add_argument('--desc_rdkit', action="store", default="True", help='use the molecular property using RDKit')
    parser.add_argument('--desc_fps', action="store", default="True", help='use the molecular fingerprints')
    parser.add_argument('--desc_cx', action="store", default="True", help='use the molecular property using ChemAxon')

    parser.add_argument('--modelType', action="store", default="Regression", help='ML model type, either <regression> or <classification>')

    parser.add_argument('--MissingValueFilter', action="store", default="True", help='remove the descriptor with a lot of missing values')
    parser.add_argument('--VarianceFilter', action="store", default="True", help='remove the descriptor with low variance')
    parser.add_argument('--L2Filter', action="store", default="True", help='feature selection using linear Lasso method')
    parser.add_argument('--FIFilter', action="store", default="True", help='feature selection using RF feature importance')

    parser.add_argument('-o', '--output', action="store", default="./results", help='the output folder')

    args = parser.parse_args()
    return args

####################################################################
######################### main function ############################
####################################################################
def main():
    ## ------------ load args ------------
    args = Args_Prepation(parser_desc='Feature selection')
    if True:
        desc_folder = args.input    #'./results'
        sep = args.delimiter    # ',' 
        colName_mid = args.colId    # 'Compound Name'
        desc_custom = True if args.desc_custom=="True" else False
        desc_rdkit = True if args.desc_rdkit=="True" else False
        desc_fps = True if args.desc_fps=="True" else False
        desc_cx = True if args.desc_cx=="True" else False
        model_type = args.modelType    # 'regression', 'classification'
        doMissingValueFilter = True if args.MissingValueFilter=="True" else False
        doVarianceFilter = True if args.VarianceFilter=="True" else False
        doL2Filter = True if args.L2Filter=="True" else False
        doFeatureImportanceFilter = True if args.FIFilter=="True" else False        
        folderPathOut = args.output    ## './results'

        ## 
        descType_list = []
        if desc_custom:
            descType_list.append('custom')
        if desc_rdkit:
            descType_list.append('rdkit')
        if desc_fps:
            descType_list.append('fingerprints')
        if desc_cx:
            descType_list.append('chemaxon')
        print(f"\tSelected descriptors types: {descType_list}")

    ## ------------ load data ------------
    import pandas as pd
    ## load the expt outcome
    dataTable_y = pd.read_csv(f'{desc_folder}/outcome_expt.csv', sep=sep)
    print(f"\tThe experiment outcome table has shape {dataTable_y.shape}")

    ## load all descriptor tables and merge together
    import copy
    dataTable_merged = copy.deepcopy(dataTable_y)
    for descType in descType_list:
        dataTable = pd.read_csv(f'{desc_folder}/descriptors_{descType}_processed.csv') 
        print(f"\tThere are total <{dataTable.shape[1]-1}> {descType} descriptors for <{dataTable.shape[0]}> molecules")
        dataTable_merged = dataTable_merged.merge(right=dataTable, on='Compound Name', how='left')
    print(f"\tThe merged data table has <{dataTable_merged.shape[0]}> molecules and <{dataTable_merged.shape[1]-1}> descriptors")

    ## get descriptors (X) and outcome (y)
    assert dataTable_merged.columns[0] == colName_mid, f"\tError! Make sure the 1st column in the desc/expt fileis <{colName_mid}>"
    X, y = dataTable_merged.iloc[:, 2:], dataTable_merged.iloc[:, 1:2]
    print(f"\tX has shape {X.shape}, y has shape {y.shape}")

    ## ------------ filters ------------
    score_table_dict = {}
    desc_drop = []
    ## remove descriptors with too many missing data
    if doMissingValueFilter:
        print(f"\tremove descriptors with too many missing data")
        json_file_imput_param = f"{desc_folder}/descriptor_imputation_params.dict"
        score_table_dict['MissingValueFilter'], desc_sele = missingValueFilter(list(X.columns), json_file_imput_param, nan_cutoff=0.2)
        X = X[desc_sele]
        for col in list(dataTable_merged.columns):
            if col not in desc_sele:
                if col not in desc_drop:
                    desc_drop.append(col)

    ## Variance-based Filter
    if doVarianceFilter:
        print(f"\tremove descriptors with too low variance")
        score_table_dict['VarianceFilter'], desc_sele = VarianceFilter(X=X, threshold=0.0001)
        X = X[desc_sele]
        for col in list(dataTable_merged.columns):
            if col not in desc_sele:
                if col not in desc_drop:
                    desc_drop.append(col)

    ## L2-based Filter
    if doL2Filter:
        print(f"\tremove descriptors with L2 regulization")
        score_table_dict['L2Filter'], desc_sele = L2_based_selection(X=X, y=y, model_type=model_type)
        for col in list(dataTable_merged.columns):
            if col not in desc_sele:
                if col not in desc_drop:
                    desc_drop.append(col)

    ## RF-based Filter
    if doFeatureImportanceFilter:
        print(f"\tremove descriptors with RF feature importance")
        score_table_dict['FIFilter'], desc_sele = RF_based_selection(X=X, y=y, model_type=model_type)
        for col in list(dataTable_merged.columns):
            if col not in desc_sele:
                if col not in desc_drop:
                    desc_drop.append(col)

    ## merge all score tables and save to csv
    score_table_merged = pd.DataFrame(columns=['Descriptor'])
    for filterType in score_table_dict:
        this_Table = score_table_dict[filterType]
        this_Table.to_csv(f"{folderPathOut}/feature_scoring_{filterType}.csv", index=False)
        
        this_Table = this_Table.rename(columns={'Select': f'select_{filterType}'})
        score_table_merged = score_table_merged.merge(right=this_Table, on='Descriptor', how='outer')
        
    fileNameOut_FSscore = f"{folderPathOut}/feature_scoring_merged.csv"
    score_table_merged.to_csv(fileNameOut_FSscore, index=False)
    print(f"\tThe merged table contains all feature selection scores is saved to <{fileNameOut_FSscore}>")

    ## clean the dataset
    cols_y = list(dataTable_y.columns)
    cols_X = [col for col in list(dataTable_merged.columns) if col not in desc_drop]
    print(f"\tThere are total <{len(cols_X)}> descriptors selected for ML modeling")
    dataTable_merged_clean = dataTable_merged[cols_y + cols_X]

    fileNameOut_4ML = f"{folderPathOut}/data_input_4_ModelBuilding.csv"
    dataTable_merged_clean.to_csv(fileNameOut_4ML, index=False)
    print(f"\tThe cleaned data table for ML model building is saved to <{fileNameOut_4ML}>")

if __name__ == '__main__':
    main()