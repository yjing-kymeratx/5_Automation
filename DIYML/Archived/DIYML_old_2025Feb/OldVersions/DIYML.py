#!/fsx/home/yjing/apps/anaconda3/env/yjing/bin python

##############################################################################################
##################################### load packages ###########################################
##############################################################################################
## avoid python warning if you are using > Python 3.11, using action="ignore"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

## load packages
import os
import copy
import time
import pickle
import chardet
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem



# from Descriptors import Desc_ChemAxon, Desc_MolFPs, Desc_RDKit
# from DataSplit import Data_Split
from ML_Dataset import Data4ML
from ML_Modeling import ML_Models

##############################################################################################
##################################### Custom Tools ###########################################
##############################################################################################
def _defineTmpFolder(folderName_tmp=None):
    if folderName_tmp is None:
        folderName_tmp = 'Tmp'
    
    folderName_tmp = os.path.join(os.getcwd(), folderName_tmp)
    os.makedirs(folderName_tmp) if not os.path.exists(folderName_tmp) else print(f'\t---->{folderName_tmp} is existing\n')
    return folderName_tmp

def _str_2_bool(myStr):
    if myStr.lower() in ['true', 't']: 
        mybool = True
    elif myStr.lower() in ['false', 'f']:
        mybool = False
    else:
        mybool = False
    return mybool

##############################################################################################
###################################### argument parser #######################################
##############################################################################################
def step_0_Args_Prepation(parser_desc):
    parser = argparse.ArgumentParser(description=parser_desc)
    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file for modeling')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')

    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Smiles', help='The column name of the compound smiles')
    parser.add_argument('--colAssay', action="store", default='IC50', help='The column names of the assay values, only 1 column is accepted')
    parser.add_argument('--yTransform', action="store", default='None', help='If the assay value needs to be taken log 10 or one-hot binary')

    parser.add_argument('--colDate', action="store", default='Created On', help='The column names of the date')
    parser.add_argument('--tmpFolder', action="store", default='Tmp', help='The column names of the assay values, separated by comma with no space')

    parser.add_argument('--desc_fps', action="store", default='True', help='<True> for calculating Molecular Fingerprints')
    parser.add_argument('--desc_cx', action="store", default='True', help='<True> for calculating ChemAxon Molecular Property')
    parser.add_argument('--desc_rdkit', action="store", default='True', help='<True> for calculating RDKit Molecular Property')

    parser.add_argument('--split', action="store", default='random', help='The method to split data into training, validation, and test. Available [random, temporal, butina]')
    parser.add_argument('--CV', action="store", default='10', help='number of Fold for split')
    parser.add_argument('--hasVal', action="store", default='True', help='If second split for a validation set')
    parser.add_argument('--randomSeed', action="store", default='666666', help='If second split for a validation set')

    parser.add_argument('--modelType', action="store", default='regression', help='a regression model or classification model')
    parser.add_argument('--model_rf', action="store", default='True', help='the RF algorithm of the ML model')
    parser.add_argument('--model_linear', action="store", default='False', help='the linear algorithm of the ML model')
    parser.add_argument('--model_svm', action="store", default='False', help='the SVM algorithm of the ML model')
    parser.add_argument('--model_mlp', action="store", default='False', help='the MLP algorithm of the ML model')
    parser.add_argument('--model_knn', action="store", default='False', help='the KNN algorithm of the ML model')
    parser.add_argument('--model_xgb', action="store", default='False', help='the XGBoosting algorithm of the ML model (Classification only)')

    parser.add_argument('--model_knnk', action="store", default="3", help='the k of KNN algorithm for the ML model')

    parser.add_argument('--HPT', action="store", default="False", help='the flag of running hyper-parameter tuning')
    return parser

##############################################################################################
###################################### prepare dataset #######################################
##############################################################################################
def step_1_Data_Preparation(args, dataName="my_ML_dataset", saveDict=True, tmpFolder='./tmp'):
    ## ---------------- Initiate an dataset object ----------------
    my_ML_Data = Data4ML(dataName=dataName)

    ## ---------------- load data from CSV file ----------------
    assert args.input is not None, f"Please provide the input file name using <-i> or <--input> option"
    # './Kymera_ADME_PK_ALL_pull.csv'
    # 'ADME AlphaLogD;Mean;AlphaLogD;(Num)'
    my_ML_Data.load_csv(fileNameIn=args.input, sep=args.delimiter, colName_mid=args.colId, colName_smi=args.colSmi, colName_activity=args.colAssay)

    ## ---------------- calculate descriptors ----------------
    desc_fps = _str_2_bool(args.desc_fps)
    desc_rdkit = _str_2_bool(args.desc_rdkit)
    desc_cx = _str_2_bool(args.desc_cx)
    my_ML_Data.calc_desc(desc_fps=desc_fps, desc_rdkit=desc_rdkit, desc_cx=desc_cx)

    ## ---------------- split data into training/validation/test set ----------------
    CV = int(args.CV)
    rng = int(args.randomSeed)
    hasVal = _str_2_bool(args.hasVal)
    my_ML_Data.train_val_test_split(split_method=args.split, CV=CV, rng=rng, hasVal=hasVal, colName_date=args.colDate)

    ## ---------------- prepare the dataset into dataframe (ready for modeling) ----------------
    my_ML_Data.prepare_dataset(desc_fps=desc_fps, desc_rdkit=desc_rdkit, desc_cx=desc_cx)

    ## ---------------- save ----------------
    if saveDict:
        fileName_dataset = f'{tmpFolder}/{dataName}.ds'
        with open(fileName_dataset, 'wb') as dsfh:
            pickle.dump(my_ML_Data, dsfh)
            print(f'\tThe data set object has been saved to {fileName_dataset}')
    
    return my_ML_Data

##############################################################################################
###################################### prepare dataset #######################################
##############################################################################################
def step_2_Data_Preprocessing(ML_dataSet, cols_x_sele, transformType='None', saveProcessor=True, tmpFolder='./tmp'):
    dataDict_ds = {}

    my_preProcessor = ML_Models.preProcessor()
    dataDict_ds['Training_X'] = my_preProcessor.PreProcess_X(ML_dataSet.X_Training, cols_sele=cols_x_sele, isTrain=True)
    dataDict_ds['Validation_X'] = my_preProcessor.PreProcess_X(ML_dataSet.X_Validation, cols_sele=cols_x_sele, isTrain=False)
    dataDict_ds['Test_X'] = my_preProcessor.PreProcess_X(ML_dataSet.X_Test, cols_sele=cols_x_sele, isTrain=False)
    
    dataDict_ds['Training_y'] = my_preProcessor.PreProcess_y(ML_dataSet.y_Training, transformType=transformType, isTrain=True)
    dataDict_ds['Validation_y'] = my_preProcessor.PreProcess_y(ML_dataSet.y_Validation, transformType=transformType, isTrain=False)
    dataDict_ds['Test_y'] = my_preProcessor.PreProcess_y(ML_dataSet.y_Test, transformType=transformType, isTrain=False)

    if saveProcessor:
        fileName_processor = f'{tmpFolder}/processor.model'
        with open(fileName_processor, 'wb') as procfh:
            pickle.dump(my_preProcessor, procfh)
            print(f'\tThe processor object has been saved to {fileName_processor}')

        fileName_procDataSet = f'{tmpFolder}/ML_dataSet_norm.ds'
        with open(fileName_procDataSet, 'wb') as dspfh:
            pickle.dump(dataDict_ds, dspfh)
            print(f'\tThe processed dataset has been saved to {fileName_procDataSet}')
    return dataDict_ds

##############################################################################################
###################################### prepare models ########################################
##############################################################################################
def step_3_Model_building(dataDict_ds, args, n_jobs=-1, saveModel=True, tmpFolder='./tmp'):
    ## get parameters from args
    rng = int(args.randomSeed)
    mType = args.modelType
    m_rf = _str_2_bool(args.model_rf)
    m_li = _str_2_bool(args.model_linear)
    m_svm = _str_2_bool(args.model_svm)
    m_mlp = _str_2_bool(args.model_mlp)
    m_knn = _str_2_bool(args.model_knn)
    m_knnk = int(args.model_knnk)
    m_xgb = _str_2_bool(args.model_xgb)
    HPT = _str_2_bool(args.HPT)

    tba = True    # retrain_by_all

    ## get model
    dict_models = {}

    if mType == 'regression':
        ## Random Forests models
        if m_rf:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'RF', rng=rng, n_jobs=n_jobs)
            dict_models['RF'] = _R_Model(dataDict_ds, sk_model, modelName=f"{mType}_RF", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)
        
        ## linear models
        if m_li:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'linear', rng=rng, n_jobs=n_jobs)            
            dict_models['linear'] = _R_Model(dataDict_ds, sk_model, modelName=f"{mType}_linear", rng=rng, n_jobs=n_jobs, HPT=False, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## SVM models
        if m_svm:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'SVM', rng=rng, n_jobs=n_jobs)            
            dict_models['SVM'] = _R_Model(dataDict_ds, sk_model, modelName=f"{mType}_SVM", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## Multi-layer perceptron models
        if m_mlp:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'MLP', rng=rng, n_jobs=n_jobs)            
            dict_models['MLP'] = _R_Model(dataDict_ds, sk_model, modelName=f"{mType}_MLP", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## K-nearest neighbor models
        if m_knn:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'KNN', rng=rng, n_jobs=n_jobs, knnk=m_knnk)            
            dict_models['KNN'] = _R_Model(dataDict_ds, sk_model, modelName=f"{mType}_KNN", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

    elif mType == 'classification': 
        ## Random Forests models
        if m_rf:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'RF', rng=rng, n_jobs=n_jobs)
            dict_models['RF'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_RF", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)
        
        ## linear models
        if m_li:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'linear', rng=rng, n_jobs=n_jobs)            
            dict_models['linear'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_linear", rng=rng, n_jobs=n_jobs, HPT=False, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## SVM models
        if m_svm:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'SVM', rng=rng, n_jobs=n_jobs)            
            dict_models['SVM'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_SVM", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## Multi-layer perceptron models
        if m_mlp:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'MLP', rng=rng, n_jobs=n_jobs)            
            dict_models['MLP'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_MLP", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

        ## K-nearest neighbor models
        if m_knn:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'KNN', rng=rng, n_jobs=n_jobs, knnk=m_knnk)            
            dict_models['KNN'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_KNN", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)
        
        ## XG-Boosting models
        if m_xgb:
            sk_model, s_space = ML_Models.select_ML_methods(mType, 'XGBoost', rng=rng, n_jobs=n_jobs)            
            dict_models['XGBoost'] = _C_Model(dataDict_ds, sk_model, modelName=f"{mType}_XGBoost", rng=rng, n_jobs=n_jobs, HPT=HPT, search_space=s_space, retrain_by_all=tba, tmpFolder=tmpFolder)

    else:
        print(f"\tError! No selected ML methods for modeling. Either <regression> or <classification>")

    if saveModel:
        os.makedirs(f'{tmpFolder}/models') if not os.path.exists(f'{tmpFolder}/models') else print(f'\t---->{tmpFolder}/models/ is existing\n')
        fileName_modelDict = f'{tmpFolder}/models/All_models.dict'
        with open(fileName_modelDict, 'wb') as mofh:
            pickle.dump(dict_models, mofh)
            print(f'\tThe Dict contains all models has been saved to {fileName_modelDict}\n') 

    return dict_models

## ----------- regression model -----------
def _R_Model(dataDict_ds, sk_model, modelName="ML_model", rng=666666, n_jobs=-1, HPT=True, search_space=None, retrain_by_all=False, saveModel=True, saveFig=True, tmpFolder='./tmp'):
    print(f"\n\t-------------------- Now building the model <{modelName}> --------------------")
    myModel = ML_Models.Regression_Model(myScikitModel=sk_model, modelName=modelName, rng=rng, n_jobs=n_jobs)
    
    ## train the model
    print(f"\t ----- Now start training the model -----")
    myModel.Train(dataDict_ds['Training_X'], dataDict_ds['Training_y'], printLog=True, HPT=HPT, search_space=search_space)
    myModel.Evaluate(dataDict_ds['Training_X'], dataDict_ds['Training_y'], ds_label='Training', printLog=True, plotResult=True)
    
    ## validate the model
    print(f"\t ----- Now start evaluating the model using validation set -----")
    myModel.Evaluate(dataDict_ds['Validation_X'], dataDict_ds['Validation_y'], ds_label='Validation', printLog=True, plotResult=True)    
    
    ## test the model
    print(f"\t ----- Now start evaluating the model using test set -----")
    myModel.Evaluate(dataDict_ds['Test_X'], dataDict_ds['Test_y'], ds_label='Test', printLog=True, plotResult=True)

    
    ## plot all
    myModel.plots['All'] = myModel._Plot_Pred_VS_Expt(dataTable=myModel.predictions, 
                                                      label_x='Prediction', label_y='Experiment', color_by='DataSet', 
                                                      diagonal=True, sideHist=False, figTitle=None) 
    ## retrain the model using all
    if retrain_by_all:
        dataDict_ds_all_X = pd.concat([dataDict_ds[f"{dl}_X"] for dl in ["Training", "Validation", "Test"]])
        dataDict_ds_all_y = pd.concat([dataDict_ds[f"{dl}_y"] for dl in ["Training", "Validation", "Test"]])
        myModel.Train(dataDict_ds_all_X, dataDict_ds_all_y, HPT=False, printLog=True)

    ## save model and figs
    saveModel = True   
    if saveModel:
        os.makedirs(f'{tmpFolder}/models') if not os.path.exists(f'{tmpFolder}/models') else print(f'\t---->{tmpFolder}/models/ is existing\n')
        fileName_modelObj = f'{tmpFolder}/models/{modelName}.DIYmodel'
        with open(fileName_modelObj, 'wb') as mofh:
            pickle.dump(myModel, mofh)
            print(f'\tThe DIYML model <{modelName}> has been saved to {fileName_modelObj}\n')        
    if saveFig:
        os.makedirs(f'{tmpFolder}/figures') if not os.path.exists(f'{tmpFolder}/figures') else print(f'\t---->{tmpFolder}/figures/ is existing\n') 
        for fig_label in myModel.plots:
            fileName_fig = f'{tmpFolder}/figures/{modelName}_{fig_label}.png'
            myModel.plots[fig_label].savefig(fileName_fig)
    return myModel

## ----------- classification model -----------
def _C_Model(dataDict_ds, sk_model, modelName="ML_model", rng=666666, n_jobs=-1, HPT=True, search_space=None, retrain_by_all=False, saveModel=True, saveFig=True, tmpFolder='./tmp'):
    print(f"\n\t-------------------- Now building the model <{modelName}> --------------------")
    myModel = ML_Models.Classification_Model(myScikitModel=sk_model, modelName=modelName, rng=rng, n_jobs=n_jobs)
    
    ## train the model
    print(f"\t ----- Now start training the model -----")
    myModel.Train(dataDict_ds['Training_X'], dataDict_ds['Training_y'], printLog=True, HPT=HPT, search_space=search_space)
    
    ## validate the model
    print(f"\t ----- Now start evaluating the model using validation set -----")
    myModel.Evaluate(dataDict_ds['Validation_X'], dataDict_ds['Validation_y'], ds_label='Validation', estCutoff=True, printLog=True, plotResult=True)
    myModel.Evaluate(dataDict_ds['Training_X'], dataDict_ds['Training_y'], ds_label='Training', estCutoff=False, printLog=True, plotResult=True)

    ## test the model
    print(f"\t ----- Now start evaluating the model using test set -----")
    myModel.Evaluate(dataDict_ds['Test_X'], dataDict_ds['Test_y'], ds_label='Test', estCutoff=False, printLog=True, plotResult=False)
    
    ## plot all
    # myModel.plots['All'] = None 

    ## retrain the model using all
    if retrain_by_all:
        dataDict_ds_all_X = pd.concat([dataDict_ds[f"{dl}_X"] for dl in ["Training", "Validation", "Test"]])
        dataDict_ds_all_y = pd.concat([dataDict_ds[f"{dl}_y"] for dl in ["Training", "Validation", "Test"]])
        myModel.Train(dataDict_ds_all_X, dataDict_ds_all_y, HPT=False, printLog=True)

    ## save model and figs    
    if saveModel:
        os.makedirs(f'{tmpFolder}/models') if not os.path.exists(f'{tmpFolder}/models') else print(f'\t---->{tmpFolder}/models/ is existing\n')
        fileName_modelObj = f'{tmpFolder}/models/{modelName}.DIYmodel'
        with open(fileName_modelObj, 'wb') as mofh:
            pickle.dump(myModel, mofh)
            print(f'\tThe DIYML model <{modelName}> has been saved to {fileName_modelObj}\n')        
    if saveFig:
        os.makedirs(f'{tmpFolder}/figures') if not os.path.exists(f'{tmpFolder}/figures') else print(f'\t---->{tmpFolder}/figures/ is existing\n') 
        for fig_label in myModel.plots:
            fileName_fig = f'{tmpFolder}/figures/{modelName}_{fig_label}.png'
            myModel.plots[fig_label].savefig(fileName_fig)
    return myModel

##############################################################################################
#################################### The main function #######################################
##############################################################################################
def main():
    beginTime = time.time()
    ## ------------------------------------- argument parser --------------------------------------
    ## parse the arguments
    parser = step_0_Args_Prepation(parser_desc='This is the script to build ML models for predicting experimental values')
    args = parser.parse_args()
    ##
    tmpFolder = _defineTmpFolder(args.tmpFolder)
    n_jobs = -1
    
    ## ------------------------------------------- run --------------------------------------------
    ## -------------- step 1 Prepere dataset --------------
    print(f"\n==>Step 1 Prepere dataset ...")
    ML_dataSet = step_1_Data_Preparation(args, dataName="ML_dataSet", saveDict=True, tmpFolder=tmpFolder)

    ## -------------- step 2: Normalize the descriptor and take log on y --------------
    print(f"\n==>Step 2 Normalize the descriptor and take log on y ...")
    cols_x_sele = [col for col in ML_dataSet.X_Training.columns if col[0:3] in ['cx_', 'rd_']]
    print(f"\tFeature normalization on {len(cols_x_sele)} columns: {cols_x_sele}")
    dataDict_ds = step_2_Data_Preprocessing(ML_dataSet, cols_x_sele, transformType=args.yTransform, saveProcessor=True, tmpFolder=tmpFolder)
    
    ## -------------- step 3 Build models --------------    
    print(f"\n==>Step 3 Build model ...")
    dataDict_model = step_3_Model_building(dataDict_ds, args, n_jobs=n_jobs, saveModel=True,  tmpFolder=tmpFolder)

if __name__ == '__main__':
    main()