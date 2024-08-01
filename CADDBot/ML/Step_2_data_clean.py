#!/usr/bin/env python

##########################################################################
######################### 1. load the packages ###########################
##########################################################################
## ignore warning msg
import warnings
warnings.filterwarnings('ignore')

import copy
import chardet
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
today = datetime.today().date().strftime('%Y-%m-%d')


from AutoML.data_clean import *
from AutoML.utility import determine_encoding



##########################################################################
####################### 2. Build custom functions ########################
##########################################################################
## ------------------------ determine_encoding ------------------------
def determine_encoding(dataFile):
    # Step 1: Open the CSV file in binary mode
    with open(dataFile, 'rb') as f:
        data = f.read()
    
    # Step 2: Detect the encoding using the chardet library
    encoding_result = chardet.detect(data)

    # Step 3: Retrieve the encoding information
    encoding = encoding_result['encoding']

    # Step 4: Print/export the detected encoding information
    # print("Detected Encoding:", encoding)
    return encoding


## ------------------------ processing data ------------------------
def extractDataFromTable(row, colName_mod=None, colName_num='col_num'):
    assert colName_mod in row, f'Cannot find <Mod> column with name <{colName_mod}!'
    assert colName_num in row, f'Cannot find <Mod> column with name {colName_num}'

    result = np.nan

    ## data columns
    if colName_mod is not None:
        if row.notna()[colName_mod] and row.notna()[colName_num]:
            if row[colName_mod] == '=':
                result = row[colName_num]
    ## comment columns
    else:
        if row.notna()[colName_num]:
            result = row[colName_num]
    return result

## calculate the mean value from a list of values
def calc_mean(value_list):
    try:
        value_list_clean = []
        for v in value_list:
            if v not in [None, np.nan, '', ' ']:
                try:
                    v_num = float(v)
                except Exception as e:
                    print(f'Error, cannot numericalize value {v}', e)
                else:
                    value_list_clean.append(v_num)
        value_ave = np.mean(value_list_clean)
    except Exception as e:
        print(f'\tCan not calulate the mean value of list {value_list}')
    else:
        value_ave = np.nan
    return value_ave

## ------------------------ hERG data ------------------------
def calc_eIC50_hERG(comments_str):
    # e.g., comments_str = '21.38% inhibition @ 10 ?M;;;'
    hERG_eIC50_list = []
    
    ## split the comments by ';'
    for cmt in comments_str.split(';'):
        try:
            ## process the comment string and calc the eIC50
            [str_inhb, str_conc] = cmt.split('@')
            inhb = float(str_inhb.split('%')[0])
            inhb = 0.1 if inhb < 0 else (99.99 if inhb > 100 else inhb)
            conc = float(str_conc.split('M')[0][:-1])
            eIC50 = conc*(100-inhb)/inhb
        except Exception as e:
            if cmt not in ['', ' ', '/']:
                print(f'\tError, cannot calc hERG eIC50 from comment data. {cmt}')
        else:
            hERG_eIC50_list.append(eIC50)

    ## calculate the mean value of eIC50
    hERG_eIC50 = calc_mean(hERG_eIC50_list)
    return hERG_eIC50





##########################################################################
####################### 3. define the main func ##########################
##########################################################################
def main():

    ## ------------------------ define the parser ------------------------
    # Create the parser
    parser = argparse.ArgumentParser(description='Test version of autoML')

    # Add arguments
    parser.add_argument('-i', '--input', type=str, default='./Data/D360_api_pull_raw.csv', help='the input file (.csv or .tsv) contains the molecules and experimental data')
    parser.add_argument('--sep', type=str, default=',', help='the delimiter in the input file to separate the column')

    parser.add_argument('--permeability', action='store_true', help='clean the permeability (MDCK WT A2B) data')  # on/off flag
    parser.add_argument('--efflux', action='store_true', help='clean the efflux ratio (MDCK MDR1) data')  # on/off flag
    
    parser.add_argument('--hERG_IC50', action='store_true', help='clean the hERG data')  # on/off flag
    parser.add_argument('--hERG_eIC50', action='store_true', help='calculate the est. hERG IC50 data')  # on/off flag

    parser.add_argument('--bioavailability', action='store_true', help='clean the F% data')  # on/off flag
    parser.add_argument('--species', type=str, default='Rat', help='the species of F% data')
    parser.add_argument('--estFa', action='store_true', help='calculate the Est.Fa data')  # on/off flag


    parser.add_argument('-o', '--output', type=str, default='./Data/D360_api_pull_clean.csv', help='the output file (.csv or .tsv) contains the cleaned data')


    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    INPUT_FILE = args.input
    SEP = args.sep
    CLEAN_PERM = args.permeability
    CLEAN_EFFLUX = args.efflux
    CLEAN_HERG = args.hERG_IC50
    CLEAN_ESTHERG = args.hERG_eIC50

    ## ------------------------ load the dataset ------------------------
    encoding = determine_encoding(INPUT_FILE)
    dataTable_raw = pd.read_csv(INPUT_FILE, sep=SEP, encoding=encoding)
    n_rows, n_cols = dataTable_raw.shape[0], dataTable_raw.shape[1]
    print(f'Reading in data table <{INPUT_FILE}> with {n_rows} rows and {n_cols} columns')

    dataTable_new = copy.deepcopy(dataTable_raw)

    ## ------------------------ clean the permeability data ------------------------
    if CLEAN_PERM:
        colName_prefix = 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s)'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'        
        dataTable_new['KT_Permeability'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

    ## ------------------------ clean the efflux data ------------------------
    if CLEAN_EFFLUX:
        colName_prefix = 'ADME MDCK (MDR1) efflux;Mean;Efflux Ratio'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'        
        dataTable_new['KT_EffluxRatio'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

    ## ------------------------ clean the hERG data ------------------------
    if CLEAN_HERG:
        colName_prefix = 'ADME Tox-manual patch hERG 34C;GMean;m-patch hERG IC50 [uM]'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'        
        dataTable_new['KT_hERG_IC50_uM'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

    if CLEAN_ESTHERG:
        colName_num = 'ADME Tox-manual patch hERG 34C;Concat;Comments'

        calc_eIC50_hERG(comments_str)
        dataTable_new['KT_hERG_eIC50_uM'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_num))


## ------------- define/select useful columns -------------
cols_dict = {
    'Mol_id': 'PROTAC_id',
    'Mol_smi': 'PROTAC_smi',
    'Mol_proj': 'KYM_Project',
    'Mol_date': 'KYM_RegDate',
    'Mol_anno': 'KYM_ExternalID',

    'Expt_Solubility_Kinetic_ugmL_Mod': 'KYM_KinSolub_ug/mL_Mod',
    'Expt_Solubility_Kinetic_ugmL_Num': 'KYM_KinSolub_ug/mL',   
    'Expt_Solubility_FASSIF_uM_Mod': 'KYM_FASSIF_Solub_uM_Mod',
    'Expt_Solubility_FASSIF_uM_Num': 'KYM_FASSIF_Solub_uM',   

    'Expt_logD_Alpha_Mod': 'KYM_logD_Mod',
    'Expt_logD_Alpha_Num': 'KYM_logD',
    'Expt_logD_ShakeFlask_Mod': '',
    'Expt_logD_ShakeFlask_Num': '',

 
    'Expt_Permeability_MDCK_WT_Mod': 'KYM_MDCKperm_Mod',
    'Expt_Permeability_MDCK_WT_Num': 'KYM_MDCKperm',
    'Expt_Permeability_MDCK_WT_Recovery': 'KYM_MDCKperm_Recovery%',

    'Expt_Efflux_MDCK_MDR1_Mod': 'KYM_EffluxRatio_Mod',
    'Expt_Efflux_MDCK_MDR1_Num': 'KYM_EffluxRatio',

    'Expt_PK_F%_Rat_PO_Mod': 'KYM_F%_10mg/kg_PO_Rat_Mod',
    'Expt_PK_F%_Rat_PO_Num': 'KYM_F%_10mg/kg_PO_Rat',
    'Expt_PK_Clobs_Rat_IV_Mod': '',
    'Expt_PK_Clobs_Rat_IV_Num': '',

    'Expt_hERG_34C_IC50_uM_Mod': 'hERG_patch_Mod',
    'Expt_hERG_34C_IC50_uM_Num': 'hERG_patch_uM',
    'Expt_hERG_34C_cmt_Mod': '',
    'Expt_hERG_34C_cmt_Num': '',
}


keep_cols = [col for col in cols_dict.values() if col not in [''] ]

dataTable_new = dataTable[keep_cols]
dataTable_new = copy.deepcopy(dataTable)


## ------------- define/select useful columns -------------

cols_dict = {}

## Permeability
prop = 'permeability'
cols_dict['permeability'] = {'Mod':'KYM_MDCKperm_Mod', 'Num':'KYM_MDCKperm'}
dataTable_new[[prop]] = dataTable_new.apply(lambda row: clean_up_permeability(row, cols_dict[prop]), axis=1)


## Efflux
prop = 'efflux'
cols_dict[prop] = {'Mod': 'KYM_EffluxRatio_Mod', 'Num': 'KYM_EffluxRatio'}
dataTable_new[[prop]] = dataTable_new.apply(lambda row: clean_up_efflux(row, cols_dict[prop]), axis=1)



## PK
Species = 'Rat'
prop = [f'F%_{Species}']   # , f'EstFa_{Species}'
cols_dict[f'F%_{Species}'] = {'Mod': 'KYM_F%_10mg/kg_PO_Rat_Mod', 'Num': 'KYM_F%_10mg/kg_PO_Rat'}
dataTable_new[[prop]] = dataTable_new.apply(lambda row: clean_up_PK(row, Species, cols_dict[prop], EstFa=False), axis=1)

cols_dict[f'Cl_{Species}'] = None





## hERG
prop = ['hERG_uM']   # , f'EstFa_{Species}'
dataTable_new[prop] = dataTable_new.apply(lambda row: clean_up_hERG(row, cols_dict[prop], eIC50=False), axis=1)

prop = ['hERG_uM', 'hERG_eIC50', 'hERG_mixedIC50', 'ambitiousData']
dataTable_new[prop] = dataTable.apply(lambda row: clean_up_hERG(row, eIC50=True), axis=1)

cols_dict['hERG_uM'] = {'Mod': 'hERG_patch_Mod', 'Num': 'hERG_patch_uM'}
cols_dict['hERG_cmt'] = None









