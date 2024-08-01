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




##########################################################################
####################### 3. define the main func ##########################
##########################################################################
def main():

    ## ------------------------ define the parser ------------------------
    # Create the parser
    parser = argparse.ArgumentParser(description='Test version of autoML')

    # Add arguments
    parser.add_argument('-i, --input', type=str, default='../DATA/Kymera.tpdecomp.data.csv', help='the input file (.csv or .tsv) contains the molecules and experimental data')
    parser.add_argument('--sep', type=str, default=',', help='the delimiter in the input file to separate the column')

    parser.add_argument('-o, --output', type=str, default='./configs/train.yml', help='the output file (.csv or .tsv) contains the cleaned')


    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    INPUT_FILE = args.input
    SEP = args.sep


    ## ------------------------ load the dataset ------------------------
    encoding = determine_encoding(INPUT_FILE)
    dataTable_raw = pd.read_csv(INPUT_FILE, sep=SEP, encoding=encoding)
    n_rows, n_cols = dataTable_raw.shape[0], dataTable_raw.shape[1]
    print(f'Reading in data table <{INPUT_FILE}> with {n_rows} rows and {n_cols} columns')



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









