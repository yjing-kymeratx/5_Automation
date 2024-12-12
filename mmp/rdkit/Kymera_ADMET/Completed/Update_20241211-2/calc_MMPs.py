# !/fsx/home/yjing/apps/anaconda3/env/yjing/bin python

##############################################################################################
##################################### load packages ###########################################
##############################################################################################
import warnings
warnings.filterwarnings('ignore')

import os
import time
import datetime

import json
import shutil
import chardet
import argparse
import subprocess

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from d360api import d360api

# dateToday = datetime.datetime.today().strftime('%Y%b%d')

## =====================================================================================   
## =================================== load argparses ==================================
## =====================================================================================
def Step_0_load_args():
    parser = argparse.ArgumentParser(description='Usage ot cxcalc_runner.py, details to be added later')
    parser.add_argument('-q', action="store", type=int, default=None, help='D360 Query ID')
    parser.add_argument('-i', action="store", default=None, help='The input file downloaded from D360')
    parser.add_argument('--colName_cid', action="store", default="Compound Name", help='The column name of mol KT ID')
    parser.add_argument('--colName_smi', action="store", default="Structure", help='The column name of SMILES')
    parser.add_argument('--colName_eid', action="store", default="External ID", help='The column name of external ID')
    parser.add_argument('--colName_prj', action="store", default="Concat;Project", help='The column name of Projects')
    parser.add_argument('--prop_dict_file', action="store", default="prop_cols_matches.json", help='The json file which specify the property of interest and the columns infomation')

    args = parser.parse_args()

    return args

##--------------------------------------------------------------
def folderChecker(my_folder='./my_folder'):
    # Check if the folder exists
    check_folder = os.path.isdir(my_folder)
    # os.path.exists(dir_outputs)
    # If the folder does not exist, create it
    if not check_folder:
        os.makedirs(my_folder)
        print(f"\tCreated folder:", my_folder)
    else:
        print(f'{my_folder} is existing')
    return my_folder

################################################################################################
############################ Step-1. download & load data from D360 ############################
################################################################################################
def dataDownload(my_query_id=3539, user_name="yjing@kymeratx.com", tokenFile='yjing_D360.token'):
    # Create API connection to the PROD server
    my_d360 = d360api(provider="https://10.3.20.47:8080")  # PROD environment
    user_name = user_name
    tokenFile = tokenFile
    
    with open(tokenFile, 'r') as ofh:
        service_token = ofh.readlines()[0]

    # Authenticate connection using service token
    print(f"\tThe D360 query ID is {my_query_id}")
    my_d360.authenticate_servicetoken(servicetoken=service_token, user=user_name)
    results = my_d360.download_query_results(query_id=my_query_id)
    return results

##--------------------------------------------------------------
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

################################################################################################
def Step_1_load_data(my_query_id=3539, dataFile=None, tmp_folder="./tmp"):
    ## count time
    beginTime = time.time()
    ## ------------------------------------------------------------------
    assert my_query_id is not None or dataFile is not None, f"\tError, both <my_query_id> and <dataFile> are None"
    if my_query_id is not None:
        print(f"\tRun D360 query on ID {my_query_id}")
        ## download data from D360 using API
        dataTableFileName = dataDownload(my_query_id=my_query_id)
        print(f'\tAll data have been downloaded in file {dataTableFileName}')

        ## move the csv file to tmp folder
        dataFile = f"{tmp_folder}/{dataTableFileName}"
        shutil.move(dataTableFileName, dataFile)
        print(f"\tMove the downloaded file {dataTableFileName} to {dataFile}")
    else:
        print(f"\tDirectly loading data from {dataFile}")

    try:
        ## determine encoding type
        encoding = determine_encoding(dataFile)
        ## read csv file
        print(f"\tNow reading csv data using <{encoding}> encoding from {dataFile}")
        dataTable = pd.read_csv(dataFile, encoding=encoding).reset_index(drop=True)
    except Exception as e:
        print(f'\tError: cannot read output file {dataFile}; error msg: {e}')
        dataTable = None
    else:
        print(f'\tThe downloaded data have data shape {dataTable.shape}')
    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"====>The step 1 costs time = %ds ................" % (costTime))
    
    return dataTable  

################################################################################################
###################### Step-2. clean up data and calculate property ############################
################################################################################################
## ------------------------------------------------------------------
def _cleanUpSmiles(smi):
    try:
        ## text processing
        if "|" in smi:
            smi = smi.split("|")[0]
        smi = smi.replace("\n", "").replace("\r", "").replace("\r\n", "")

        ## rdkit checking
        mol = Chem.MolFromSmiles(smi)
        smi_rdkit = Chem.MolToSmiles(mol)
    except:
        smi_rdkit = np.nan
    return smi_rdkit

## ------------------------------------------------------------------
def CheckThePropertyDataStats(dataTable, col_prop_prefix, propName):
    col_mod, col_num = f"{col_prop_prefix}(Mod)", f"{col_prop_prefix}(Num)"
    if (col_mod in dataTable) and (col_num in dataTable):
        cond_1 = (dataTable[col_mod]=='=')
        cond_2 = (dataTable[col_num].notna())
        # print(dataTable[cond_1].shape, dataTable[cond_2].shape)
        data_size_available = dataTable[cond_1 & cond_2].shape[0]
        print(f"\tThere are total {data_size_available} existing data for {propName}")
        passCheck = True
    else:
        print(f"\tWarning! The column {col_prop_prefix}(Mod)/(Num) is not in the table.")
        passCheck = False
    return passCheck

## ------------------------------------------------------------------
def clean_up_prop_data(row, col_prop_prefix, propName):
    colName_mod = f"{col_prop_prefix}(Mod)"
    colName_num = f"{col_prop_prefix}(Num)"

    if row[colName_mod] == '=' and row.notna()[colName_num]:
        result = row[colName_num] 
    else:
        result = np.nan
    return result

## ----------------------------- F% and EstFa -------------------------------------
def rm_elacridar_records(row, col_perctgF='Bioavailability', col_vehicle='ADME PK;Concat;Vehicle'):
    result = row[col_perctgF]
    if row.notna()[col_vehicle]:
        if 'elacridar' in row[col_vehicle]:
            result = np.nan
            print(f"\t------>change from {row[col_perctgF]} to np.nan, {row[col_vehicle]}")
    return result

def calc_EstFa_fromAdm(PKF_PO, Clobs_IV, Species='Rat'):
    dict_IV_ratio = {'Rat': 90, 'Mouse': 70, 'Dog': 30, 'Monkey': 44}    
    try:
        estfa = (PKF_PO/100)/(1-(Clobs_IV/dict_IV_ratio[Species]))
    except Exception as e:
        estfa = np.nan
    return estfa

def calc_EstFa(row, colName_pctF, colName_Clobs, Species='Rat'):
    try:
        pctgF_PO, Clobs_IV = row[colName_pctgF], row[colName_Clobs]
    except Exception as e:
        # print(f"\tWarning! Cannot get data for this row from column <{colName_pctgF}> or <{colName_Clobs}>")
        result = np.nan
    else:
        result = calc_EstFa(pctgF_PO, Clobs_IV, Species=Species)
    return result

## ----------------------------- hERG -------------------------------------
def calc_mean(value_list):
    value_list_clean = []
    for v in value_list:
        if v not in [None, np.nan, '', ' ']:
            try:
                v_num = float(v)
            except Exception as e:
                print(f'\tError, cannot numericalize value {v}', e)
            else:
                value_list_clean.append(v_num)
    return np.mean(value_list_clean)

def calc_eIC50_hERG_from_cmt(comments_str):
    # e.g., comments_str = '21.38% inhibition @ 10 ?M' or '11.17 inhibition @ 3 ?M'
    try:
        [str_inhb, str_conc] = comments_str.split('@')

        if '%' in str_inhb:
            inhb = str_inhb.split('%')[0]
        elif 'inhibit' in str_inhb:
            inhb = str_inhb.split('inhibit')[0]
        else:
            inhb = 'N/A'
        
        try:
            inhb = float(inhb)
        except:
            eIC50 = None
        else:
            inhb = 0.1 if inhb < 0 else (99.99 if inhb > 100 else inhb)
            conc = float(str_conc.split('M')[0][:-1])
            eIC50 = conc*(100-inhb)/inhb
            
    except Exception as e:
        eIC50 = None
        if comments_str not in [' ', '/']:
            print(f'\tError, cannot calc hERG eIC50 from comment data. {comments_str}')
    return eIC50

def calc_hERG_eIC50(row, col_hERG_cmts):
    if col_hERG_cmts in row:
        if row.notna()[col_hERG_cmts]:
            hERG_eIC50_list = []
            for cmnt in row[col_hERG_cmts].split(';'):
                this_eIC50 = calc_eIC50_hERG_from_cmt(cmnt)
                hERG_eIC50_list.append(this_eIC50)
            hERG_eIC50 = calc_mean(hERG_eIC50_list)
        else:
            result = np.nan
            # print(f"\tNo data in this row for column <{col_hERG_cmts}>")
    else:
        result = np.nan
        print(f"\tColumn <{col_hERG_cmts}> is not in the Table")

def calc_hERG_mIC50(row, col_hERG_IC50, col_hERG_eIC50):
    if row.notna()[col_hERG_IC50]:
        result = row[col_hERG_IC50]
    elif row.notna()[col_hERG_eIC50]:
        result = row[col_hERG_eIC50]
    else:
        result = np.nan
    return result

################################################################################################
def Step_2_clean_data(dataTable, dict_prop_cols, colName_mid, colName_smi, tmp_folder="./tmp"):
    ## count time
    beginTime = time.time()
    ## ------------------------------------------------------------------
    dataTable[f"{colName_smi}_raw"] = dataTable[colName_smi].apply(lambda x: x)
    dataTable[colName_smi] = dataTable[colName_smi].apply(_cleanUpSmiles)
    dataTable = dataTable.dropna(subset=[colName_mid, colName_smi]).reset_index(drop=True)
    print(f'\tThere are total {dataTable.shape[0]} molecules with valid SMILES<{colName_smi}>')

    ## ------------------------------------------------------------------
    for prop in dict_prop_cols:
        passCheck = CheckThePropertyDataStats(dataTable, col_prop_prefix=dict_prop_cols[prop], propName=prop)
        if passCheck:
            dataTable[prop] = dataTable.apply(lambda row: clean_up_prop_data(row, col_prop_prefix=dict_prop_cols[prop], propName=prop), axis=1)

        ## remove the 'elacridar' records
        if prop == 'Bioavailability':
            print(f"\t    The num rows with cleaned {prop} data (raw) is:", str(dataTable[dataTable[prop].notna()].shape[0]))
            dataTable[prop] = dataTable.apply(lambda row: rm_elacridar_records(row, col_perctgF=prop, col_vehicle='ADME PK;Concat;Vehicle'), axis=1)
            print(f"\t    The num rows with cleaned {prop} data (no elacridar) is:", str(dataTable[dataTable[prop].notna()].shape[0]))

        ## calc estFa
        if prop == 'estFa':
            dataTable[prop] = dataTable.apply(lambda row: calc_EstFa(row, 'Bioavailability', 'Cl_obs', Species='Rat'), axis=1)

        ## calc hERG eIC50
        if prop == 'hERG_eIC50':
            dataTable[prop] = dataTable.apply(lambda row: calc_hERG_eIC50(row, dict_prop_cols[prop]), axis=1)
        
        if prop == 'hERG_mixedIC50':
            dataTable[prop] = dataTable.apply(lambda row: calc_hERG_mIC50(row, 'hERG_IC50', 'hERG_eIC50'), axis=1)

        ## rename MW
        if prop == 'MW':
            dataTable[prop] = dataTable[dict_prop_cols[prop]].apply(lambda x: x)

        ## report
        print(f"\t    The num rows with cleaned {prop} data is:", str(dataTable[dataTable[prop].notna()].shape[0]))
    
    ## ------------------------------------------------------------------
    colNames_basic = [colName_mid, colName_smi]
    colName_props = list(dict_prop_cols.keys())
    dataTable_4_mmp = dataTable[colNames_basic + colName_props]

    dateToday = datetime.datetime.today().strftime('%Y%b%d')
    dataTable_4_mmp.to_csv(f'{tmp_folder}/Data_4_MMP_{dateToday}.csv', index=False)
    print(f'\tThe cleaned dataTable have data shape {dataTable_4_mmp.shape}')

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"====>The step 2 costs time = %ds ................" % (costTime))
    return dataTable_4_mmp 

################################################################################################
################################### Step-3. MMPs analysis ######################################
################################################################################################
## ---------------- prepare the Smiles file and property file ----------------
def prep_smi_file(dataTable, colName_prop_list, colName_mid='Compound Name', colName_smi='Structure', output_folder='./results'):
    print(f"\tNow starting preparing the SMILES file and property CSV file for mmpdb ...")
    
    ## the SMILES file for fragmentation
    file_smi = f'{output_folder}/Compounds_All.smi'
    file_prop_csv = f'{output_folder}/Property_All.csv'
    delimiter=' '
    ##
    data_dict_prop = {}
    with open(file_smi, "w") as output_file:
        # output_file.write(f'SMILES{delimiter_smi}ID' + "\n")
        for idx in dataTable.index:
            mol_id = dataTable[colName_mid][idx]
            mol_smi = dataTable[colName_smi][idx]

            ## prepare the SMILES output
            this_line = f'{mol_smi}{delimiter}{mol_id}'
            output_file.write(this_line + "\n")  # Add a newline character after each string

            ## prepare the property CSV output as dict
            data_dict_prop[idx] = {}
            data_dict_prop[idx]['ID'] = mol_id

            for prop_name in colName_prop_list:
                try:
                    if dataTable_raw[prop_name].notna()[idx]:
                        mol_prop = float(dataTable_raw[prop_name][idx])
                    else:
                        mol_prop = "*"
                except Exception as e:
                    data_dict_prop[idx][prop_name] = "*"
                    # print(f'\tThis mol {mol_id} does not have a proper property value: {e}')
                else:
                    data_dict_prop[idx][prop_name] = mol_prop
        
    print(f'\tThe SMILES strings have been saved into .smi file: {file_smi}')
        
    ## save the csv results
    data_table_prop = pd.DataFrame.from_dict(data_dict_prop).T
    data_table_prop.to_csv(file_prop_csv, index=False, sep=delimiter)
    print(f'\tThe property data ({data_table_prop.shape}) have been saved into .csv file: {file_smi}')
    # data_table_prop.head(3)
    return file_smi, file_prop_csv
        
## ---------------- basic cmd run ----------------
def run_cmd(commandLine):
    # beginTime = time.time()

    # Use subprocess to execute the command
    process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # costTime = time.time()-beginTime
    # print(f"\tThis command costs time = %ds ................" % (costTime))
    return (output, error)

################################################################################################
def Step_3_mmp_analysis(dataTable, dict_prop_cols, colName_mid='Compound Name', colName_smi='Structure', output_folder='./results'):
    ## count time
    beginTime = time.time()
    ## ------------------------------------------------------------------    
    ## prepare the Smiles file and property file
    colName_prop_list = list(dict_prop_cols)
    file_smi, file_prop_csv = prep_smi_file(dataTable, colName_prop_list, colName_mid, colName_smi, output_folder)

    ## ------------------------------------------------------------------
    ## Fragment the SMILES
    file_fragdb = f'{output_folder}/Compounds_All.fragdb'
    commandLine_1 = ['mmpdb', 'fragment', file_smi, '-o', file_fragdb]
    (output_1, error_1) = run_cmd(commandLine_1)
    print(f'\tThe fragmentation is completed and saved into file {file_fragdb}')

    ## ------------------------------------------------------------------
    ## Indexing to find the MMPs in the fragment file & Load the activity/property data
    file_mmpdb = f'{output_folder}/Compounds_All.mmpdb'
    commandLine_2 = ['mmpdb', 'index', file_fragdb, '-o', file_mmpdb, '--properties', file_prop_csv]
    (output_2, error_2) = run_cmd(commandLine_2)
    print(f'\tThe indexing/mmp generation is completed and saved into file {file_mmpdb}')

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"====>The step 3 costs time = %ds ................" % (costTime))

    return file_mmpdb

################################################################################################
######################################## main ##################################################
################################################################################################
def main():
    ## ------------------------------------------------------------------
    print(f"==> Step 0: load the parameters ... ")
    args = Step_0_load_args()

    ## ------------------------------------------------------------------
    my_query_id = args.q    # 3539
    dataFile = args.i    # None

    colName_mid = args.colName_cid    # 'Compound Name'
    colName_smi = args.colName_smi    # 'Structure' or 'Smiles'
    # colName_proj = args.colName_prj    # 'Concat;Project'
    # colName_eid = args.colName_eid    # 'Concat;External Id'

    # Reading JSON data from a file
    prop_dict_file = args.prop_dict_file
    print(prop_dict_file)
    with open(prop_dict_file, 'r') as infile:
        dict_prop_cols = json.load(infile)
    '''
    dict_prop_cols = {
        'Permeability': 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);', 
        'Efflux': 'ADME MDCK (MDR1) efflux;Mean;Efflux Ratio;', 
        'Bioavailability': 'ADME PK;Mean;F %;Dose: 10.000 (mg/kg);Route of Administration: PO;Species: Rat;', 
        'Cl_obs': 'Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: 2.000 (mg/kg);Route of Administration: IV;Species: Rat;',
        'hERG_IC50': 'ADME Tox-manual patch hERG 34C;GMean;m-patch hERG IC50 [uM];',
        'hERG_eIC50': 'ADME Tox-manual patch hERG 34C;Concat;Comments',
        'hERG_mixedIC50': 'Not Availale',
        'estFa': 'Not Availale',
        'MW': 'Molecular Weight',
        'bpKa1': 'in Silico PhysChem Property;Mean;Corr_ChemAxon_bpKa1;',
        'logD': 'in Silico PhysChem Property;Mean;Kymera ClogD (v1);', 
        }    
    '''

    ## ------------------------------------------------------------------
    ## create tmp folder
    tmp_folder = folderChecker(f"./tmp")
    output_folder = folderChecker(f"./results")
    
    ## ------------------------------------------------------------------
    #### Step-1. download & load data from D360
    print(f"==> Step 1: download & load data from D360 ...")
    dataTable = Step_1_load_data(my_query_id, dataFile, tmp_folder)
    # dataTable = pd.read_csv(f"./tmp/D360_dataset_q_id3539_111224_0120.csv").reset_index(drop=True)
    # dataTable.head(3)

    ## ------------------------------------------------------------------
    #### Step-2. clean up data and calculate property
    print(f"==> Step 2: clean up data and calculate new property ...")
    dataTable_4_mmp = Step_2_clean_data(dataTable, dict_prop_cols, colName_mid, colName_smi, tmp_folder)
    # dataTable_4_mmp.head(3)
    
    ## ------------------------------------------------------------------
    #### Step-3. MMPs analysis
    print(f"==> Step 3: run MMP analysis using mmpdb ...")
    file_mmpdb = Step_3_mmp_analysis(dataTable_4_mmp, dict_prop_cols, colName_mid, colName_smi, output_folder)

if __name__ == '__main__':
    main()