#!/usr/bin/env python

##########################################################################
######################### 1. load the packages ###########################
##########################################################################
## ignore warning msg
import warnings
warnings.filterwarnings('ignore')

import os
import copy
import chardet
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
today = datetime.today().date().strftime('%Y-%m-%d')


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


## clean up the projects
def CleanUpProj(projects_list_string):
    # projects_list = projects_list_string.split(';')
    # projects_list

    # project_main = ['IRAK4', 'STAT-6', 'IRF5', 'TYK2', 'MGD', 'CDK2', 'MK2']
    # new = []

    # for i in range(len(project_main)):
    #     proj = project_main[i]

    return projects_list_string



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
def calc_eIC50_hERG_from_cmt(comments_str):
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

def extractDataFromTable_hERGeIC50(row, colName_num):
    assert colName_num in row, f'Cannot find <Mod> column with name <{colName_num}!'
    if row.notna()[colName_num]:
        comments_str = row[colName_num]
        eIC50 = calc_eIC50_hERG_from_cmt(comments_str)
    else:
        eIC50 = np.nan
    return eIC50



## ------------------------ Bioavailability data ------------------------
def determine_F_dose(Species):
    dict_PK_param = {
        'Rat': {'Dose_PO':'10.000 (mg/kg)', 'Dose_IV':'2.000 (mg/kg)', 'ratio':90},
        'Mouse': {'Dose_PO':'10.000 (mg/kg)', 'Dose_IV':'2.000 (mg/kg)', 'ratio':70},
        'Dog': {'Dose_PO':'3.000 (mg/kg)', 'Dose_IV':'0.500 (mg/kg)', 'ratio':30}, 
        'Monkey': {'Dose_PO':'3.000 (mg/kg)', 'Dose_IV':'0.500 (mg/kg)', 'ratio':44}}
    try:
        dose_species = dict_PK_param[Species]

    except Exception as e:
        print(f"The species {Species} is not in {dict_PK_param.keys()}")
        dose_PO, dose_IV, ratio_PI = None, None, None
    else:
        dose_PO, dose_IV, ratio_PI = dose_species['Dose_PO'], dose_species['Dose_IV'], dose_species['ratio']
    return dose_PO, dose_IV, ratio_PI


def calc_EstFa(PKF_PO, Clobs_IV, ratio):
    try:
        estfa = (PKF_PO/100)/(1-(Clobs_IV/ratio))
    except Exception as e:
        estfa = np.nan
    return estfa

##########################################################################
####################### 3. define the main func ##########################
##########################################################################

'''
"Compound Name","Structure","Concat;Project","Concat;External Id","Created On",

"ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)",
"ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)",
"ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Mod)",
"ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Num)",
"ADME MDCK(WT) Permeability;Concat;Comments",
"ADME MDCK(WT) Permeability;Concat;Run Date",
"ADME MDCK(WT) Permeability;Mean;A to B Recovery (%)",
"ADME MDCK(WT) Permeability;Mean;B to A Recovery (%)",

"ADME MDCK (MDR1) efflux;Mean;A to B Papp (10^-6 cm/s);(Mod)",
"ADME MDCK (MDR1) efflux;Mean;A to B Papp (10^-6 cm/s);(Num)",
"ADME MDCK (MDR1) efflux;Mean;B to A Papp (10^-6 cm/s);(Mod)",
"ADME MDCK (MDR1) efflux;Mean;B to A Papp (10^-6 cm/s);(Num)",
"ADME MDCK (MDR1) efflux;Concat;Comments",
"ADME MDCK (MDR1) efflux;Mean;Efflux Ratio;(Mod)",
"ADME MDCK (MDR1) efflux;Mean;Efflux Ratio;(Num)",
"ADME MDCK (MDR1) efflux;Concat;Run Date",
"ADME MDCK (MDR1) efflux;Mean;A to B Recovery (%)",
"ADME MDCK (MDR1) efflux;Mean;B to A Recovery (%)",

"ADME PK;Mean;F %;Dose: 10.000 (mg/kg);Route of Administration: PO;Species: Rat;(Mod)",
"ADME PK;Mean;F %;Dose: 10.000 (mg/kg);Route of Administration: PO;Species: Rat;(Num)",
"Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: 2.000 (mg/kg);Route of Administration: IV;Species: Rat;(Mod)",
"Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: 2.000 (mg/kg);Route of Administration: IV;Species: Rat;(Num)",

"ADME Tox-manual patch hERG 34C;Mean;Average % of hERG inhibition;(Mod)",
"ADME Tox-manual patch hERG 34C;Mean;Average % of hERG inhibition;(Num)",
"ADME Tox-manual patch hERG 34C;Concat;Comments",
"ADME Tox-manual patch hERG 34C;Mean;Concentration (uM);(Mod)",
"ADME Tox-manual patch hERG 34C;Mean;Concentration (uM);(Num)",
"ADME Tox-manual patch hERG 34C;Concat;Date run",
"ADME Tox-manual patch hERG 34C;GMean;m-patch hERG IC50 [uM];(Mod)",
"ADME Tox-manual patch hERG 34C;GMean;m-patch hERG IC50 [uM];(Num)",
"ADME Tox-manual patch hERG 34C;Mean;SD;(Mod)",
"ADME Tox-manual patch hERG 34C;Mean;SD;(Num)",
"Marked"
'''

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
    # parser.add_argument('--species', type=str, default='Rat', help='the species of F% data')
    parser.add_argument('--estFa', action='store_true', help='calculate the Est.Fa data')  # on/off flag


    # parser.add_argument('-o', '--output', type=str, default='./Data/D360_api_pull_clean.csv', help='the output file (.csv or .tsv) contains the cleaned data')


    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    INPUT_FILE = args.input
    SEP = args.sep
    CLEAN_PERM = args.permeability
    CLEAN_EFFLUX = args.efflux
    CLEAN_HERG = args.hERG_IC50
    CLEAN_EHERG = args.hERG_eIC50
    CLEAN_PCTF = args.bioavailability
    CLEAN_ESTFA = args.estFa
    # SPECIES = args.species
    SPECIES = 'Rat'

    ## ------------------------ load the dataset ------------------------
    encoding = determine_encoding(INPUT_FILE)
    dataTable_raw = pd.read_csv(INPUT_FILE, sep=SEP, encoding=encoding)
    n_rows, n_cols = dataTable_raw.shape[0], dataTable_raw.shape[1]
    print(f'Reading in data table <{INPUT_FILE}> with {n_rows} rows and {n_cols} columns')

    dataTable_new = copy.deepcopy(dataTable_raw)
    dataDir = './Data'
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)    

    ## ------------------------ get the basic molecular data ------------------------
    mol_info_cols = ["Compound Name", "Structure", "Concat;Project", "Concat;External Id", "Created On"]
    mol_info_cols_copy = copy.deepcopy(mol_info_cols)
    colName_mid, colName_smiles = mol_info_cols[0], mol_info_cols[1]
    assert colName_mid in dataTable_new.columns, f"The Molecular ID column {colName_mid} is not availabe in the data"
    assert colName_mid in dataTable_new.columns, f"The Molecular structure column {colName_smiles} is not availabe in the data"
    for mol_info_col in mol_info_cols:
        if mol_info_col not in [colName_mid, colName_smiles]:
            if mol_info_col not in dataTable_new.columns:
                print(f'Warnning! The column {mol_info_col} is not in the data table!')
                mol_info_cols_copy.remove(mol_info_col)
    
    ## save
    dataTable_basic = dataTable_new[mol_info_cols]
    dataTable_basic.to_csv(f'{dataDir}/Compound_info.csv', index=False) 


    ## ------------------------ clean the permeability data ------------------------
    if CLEAN_PERM:
        colName_prefix = 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s)'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'

        colName_new = 'KT_Permeability'
        dataTable_new[colName_new] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    ## ------------------------ clean the efflux data ------------------------
    if CLEAN_EFFLUX:
        colName_prefix = 'ADME MDCK (MDR1) efflux;Mean;Efflux Ratio'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'

        colName_new = 'KT_EffluxRatio'     
        dataTable_new[colName_new] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    ## ------------------------ clean the hERG data ------------------------
    if CLEAN_HERG:
        colName_prefix = 'ADME Tox-manual patch hERG 34C;GMean;m-patch hERG IC50 [uM]'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'        
        
        colName_new = 'KT_hERG_IC50_uM'
        dataTable_new['KT_hERG_IC50_uM'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    if CLEAN_EHERG:
        colName_num = 'ADME Tox-manual patch hERG 34C;Concat;Comments'

        colName_new = 'KT_hERG_eIC50_uM'        
        dataTable_new[colName_new] = dataTable_new[colName_num].apply(lambda row: extractDataFromTable_hERGeIC50(row, colName_num))
        
        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)
    
    if CLEAN_HERG and CLEAN_EHERG:
        colName_new = 'KT_hERG_mixIC50_uM'
        dataTable_new[colName_new] = np.where(dataTable_new['KT_hERG_IC50_uM'].notna(), dataTable_new['KT_hERG_IC50_uM'], dataTable_new['KT_hERG_eIC50_uM'])

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    ## ------------------------ clean the hERG data ------------------------
    if CLEAN_PCTF or CLEAN_ESTFA:
        dose_PO, dose_IV, ratio_PI = determine_F_dose(SPECIES)
        colName_prefix = f'ADME PK;Mean;F %;Dose: {dose_PO};Route of Administration: PO;Species: {SPECIES}'
        colName_mod, colName_num = colName_prefix + ';(Mod)', colName_prefix + ';(Num)'

        colName_new = f'KT_PctF_{SPECIES}'
        dataTable_new[colName_new] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod, colName_num))

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    if CLEAN_ESTFA:
        colName_prefix_Cl = f'Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: {dose_IV};Route of Administration: IV;Species: {SPECIES}'
        colName_mod_Cl, colName_num_Cl = colName_prefix_Cl + ';(Mod)', colName_prefix_Cl + ';(Num)'
        dataTable_new[f'KT_Clobs_{SPECIES}'] = dataTable_new.apply(lambda row: extractDataFromTable(row, colName_mod_Cl, colName_num_Cl))
        
        colName_new = f'KT_EstFa_{SPECIES}'
        dataTable_new[colName_new] = dataTable_new.apply(lambda row: calc_EstFa(row[f'KT_PctF_{SPECIES}'], row[f'KT_Clobs_{SPECIES}'], ratio_PI))

        ## save
        dataTable_new[[colName_mid, colName_new]].to_csv(f'{dataDir}/{colName_new}.csv', index=False)

    
##########################################################################
if(__name__ == "__main__"):
    main()



    














