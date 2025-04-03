'''
tbd
'''
####################################################################
########################## Tools ###################################
####################################################################
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file')
    parser.add_argument('-p', '--parameters', action="store", default='./Parameters.csv', help='The csv file contains all the parameters of model building')
    # parser.add_argument('-o', '--outputFolder', action="store", default="./Results", help='save the modeling performance file')
    args = parser.parse_args()
    return args
##
def Load_Parameter_from_CSV(fileNameParam, true_label_list=['TRUE', 'True', 'true', 'YES', 'Yes', 'yes']):
    import pandas as pd
    print(f"\tLoad parameter files from <{fileNameParam}>")
    dataTable_params = pd.read_csv(f"{fileNameParam}")
    ParameterDict = {}
    for idx in dataTable_params.index:
        param_name = dataTable_params['ParameterName'][idx]
        if param_name not in ParameterDict:
            if dataTable_params['ParameterValue'].notna()[idx]:
                ParameterDict[param_name] = dataTable_params['ParameterValue'][idx]
            else:
                ParameterDict[param_name] = "None"

    ## deliminator
    if ParameterDict['delimiter'].lower() in ['comma']:
        ParameterDict['delimiter'] = ','
    elif ParameterDict['delimiter'].lower() in ['tab']:
        ParameterDict['delimiter'] = '\t'
    else:
        ParameterDict['delimiter'] = ','
    
    ## calculator parameters
    ParameterDict['desc_calc_param'] = {}
    ParameterDict['desc_calc_param']['rd_physChem'] = True if ParameterDict['rd_physChem'] in true_label_list else False
    ParameterDict['desc_calc_param']['rd_subStr'] = True if ParameterDict['rd_subStr'] in true_label_list else False
    ParameterDict['desc_calc_param']['rd_clean'] = True if ParameterDict['rd_clean'] in true_label_list else False
    ParameterDict['desc_calc_param']['fp_radius'] = int(ParameterDict['fp_radius'])
    ParameterDict['desc_calc_param']['fp_nBits'] = int(ParameterDict['fp_nBits'])
    ParameterDict['desc_calc_param']['cx_version'] = ParameterDict['cx_version']
    ParameterDict['desc_calc_param']['cx_desc'] = ParameterDict['cx_desc']

    ##
    ParameterDict['ml_method_list'] = []
    for ml_key in ['linear', 'rf', 'svm', 'mlp', 'knn']:
        if ParameterDict[ml_key] in true_label_list:
            ParameterDict['ml_method_list'].append(ml_key)

    
    ##
    ParameterDict['hasVal'] = "Yes"
    return ParameterDict


####################################################################
######################### main function ############################
####################################################################
def autoML():    
    ########################## load args ########################
    args = Args_Prepation(parser_desc='AutoML: automatic ML model building.')
    fileNameIn = args.input    
    true_label_list = ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes']
    ParameterDict = Load_Parameter_from_CSV(args.parameters, true_label_list)

    ########################## load params for desc calc ######################## 
    ## step-1
    detect_encoding = True if ParameterDict['detectEncoding'] in true_label_list else False
    sep = ParameterDict['delimiter']
    colName_mid = ParameterDict['colId']
    colName_smi = ParameterDict['colSmi']
    print('********************', colName_smi)
    colName_expt = ParameterDict['colAssay']
    colName_expt_operator = None if ParameterDict['colAssayMod'] == 'None' else ParameterDict['colAssayMod']
    ## step-2
    split_method = ParameterDict['split']
    colName_date = ParameterDict['colDate'] if split_method == "temporal" else None
    CV = int(ParameterDict['trainTestRatio']) + 1
    rng = int(ParameterDict['randomSeed'])
    hasVal = True if ParameterDict['hasVal'] in true_label_list else False
    ## step-3
    desc_rdkit = True if ParameterDict['desc_rdkit'] in true_label_list else False
    desc_fps = True if ParameterDict['desc_fps'] in true_label_list else False
    desc_cx = True if ParameterDict['desc_cx'] in true_label_list else False
    desc_calc_param = ParameterDict['desc_calc_param']
    do_norm = True if ParameterDict['normalization'] in true_label_list else False
    do_imputation = True if ParameterDict['imputation'] in true_label_list else False
    colName_custom_desc = ParameterDict['colPreCalcDesc']
    ## step-4
    modelType = ParameterDict['modelType']
    doMissingValueFilter = True if ParameterDict['MissingValueFilter'] in true_label_list else False
    doVarianceFilter = True if ParameterDict['VarianceFilter'] in true_label_list else False
    doL2Filter = True if ParameterDict['L2Filter'] in true_label_list else False
    doFeatureImportanceFilter = True if ParameterDict['FIFilter'] in true_label_list else False
    ## step-5
    ml_method_list = ParameterDict['ml_method_list']
    doHPT = True if ParameterDict['doHPT'] in true_label_list else False
    logy = True if ParameterDict['logy'] in true_label_list else False
    logy = True if ParameterDict['logy'] in true_label_list else False
    n_jobs = int(ParameterDict['njobs'])


    ## --------------------------------------------------------

    ######################## model building ############################
    import os, sys
    sys.path.append(os.getcwd())    # './'
    import DataPrep, DataSplit, DescGen, FeatSele, ModelBuilding

    ## ----------------- step-1 -----------------
    filePathOut_clean, filePathOut_y = DataPrep.run_script(fileNameIn=fileNameIn, sep=sep, detect_encoding=detect_encoding, 
                                                           colName_mid=colName_mid, colName_smi=colName_smi, colName_expt=colName_expt, 
                                                           colName_expt_operator=colName_expt_operator)
    print(f"Step-1 done! The cleaned file has been saved to <{filePathOut_clean}>; the y data has been saved to <{filePathOut_y}>")

    ## ----------------- step-2 -----------------
    filePathOut_split = DataSplit.run_script(fileNameIn=filePathOut_clean, sep=sep, colName_mid=colName_mid, colName_smi=colName_smi, 
                                             split_method=split_method, colName_date=colName_date, CV=CV, rng=rng, hasVal=hasVal)
    print(f"Step-2 done! The split data has been saved to <{filePathOut_split}>")

    ## ----------------- step-3 -----------------
    file_Output_S3 = DescGen.run_script(fileNameIn=filePathOut_clean, sep=sep, colName_mid=colName_mid, colName_smi=colName_smi, 
                                        desc_rdkit=desc_rdkit, desc_fps=desc_fps, desc_cx=desc_cx, 
                                        desc_calc_param=desc_calc_param, do_norm=do_norm, do_imputation=do_imputation, 
                                        colName_custom_desc=colName_custom_desc)
    filePathOut_Desc, json_file_calc_param, json_file_norm_param, json_file_imput_param = file_Output_S3[0], file_Output_S3[1], file_Output_S3[2], file_Output_S3[3]
    print(f"Step-3 done! The descriptor data has been saved to <{filePathOut_Desc}>")
    print(f"             The calculation params json data has been saved to <{json_file_calc_param}>")
    print(f"             The normalization params json data has been saved to <{json_file_norm_param}>")
    print(f"             The imputation params json data has been saved to <{json_file_imput_param}>")


    ## ----------------- step-4 -----------------


    colName_split = 'Split'
    filePathOut_4ML = FeatSele.run_script(input_X=filePathOut_Desc, input_y=filePathOut_y, input_split=filePathOut_split, sep=sep, 
                                          colName_mid=colName_mid, colName_split=colName_split, colName_expt=colName_expt,
                                          modelType=modelType,
                                          doMissingValueFilter=doMissingValueFilter, 
                                          doVarianceFilter=doVarianceFilter, 
                                          doL2Filter=doL2Filter, 
                                          doFeatureImportanceFilter=doFeatureImportanceFilter, 
                                          json_file_imput_param=json_file_imput_param)
    print(f"Step-4 done! The data 4 ML has been saved to <{filePathOut_4ML}>")
    
    ## ----------------- step-5 -----------------
    ##
    import json
    with open(json_file_calc_param, 'r') as ifh_dc:
        desc_calc_param = json.load(ifh_dc)
    with open(json_file_norm_param, 'r') as ifh_dn:
        desc_norm_param = json.load(ifh_dn)
    with open(json_file_imput_param, 'r') as ifh_di:
        desc_impu_param = json.load(ifh_di)

    ModelBuilding.runScript(fileNameIn=filePathOut_4ML, sep=sep, colName_mid=colName_mid, colName_split=colName_split, colName_y=colName_expt, modelType=modelType, ml_method_list=ml_method_list, 
                            rng=rng, n_jobs=n_jobs, logy=logy, doHPT=doHPT, desc_calc_param=desc_calc_param, desc_norm_param=desc_norm_param, desc_impu_param=desc_impu_param)
    print(f"Step-5 done!")

def main():
    print(f"----------- Start -----------")
    import time
    begin_time = time.time()
    autoML()
    print(f"----------- Total time: {round(time.time()-begin_time, 2)} sec -----------")

if __name__ == '__main__':
    main()