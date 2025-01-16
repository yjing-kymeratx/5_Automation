# !/fsx/home/yjing/apps/anaconda3/env/yjing/bin python
# calc_MMPs.py
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
import argparse
import subprocess

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from d360api import d360api

# dateToday = datetime.datetime.today().strftime('%Y%b%d')

##############################################################################################
##################################### Custom Tools ###########################################
##############################################################################################
def _folderChecker(my_folder='./my_folder'):
    ## ------- simply clean up the folder path -------
    if my_folder is None:
        my_folder='./tmp'
    elif '/' not in my_folder:
        my_folder = os.path.join(os.getcwd(), my_folder)

    ## ------- Check if the folder exists -------
    check_folder = os.path.isdir(my_folder)
    # os.path.exists(dir_outputs)
    # If the folder does not exist, create it
    if not check_folder:
        os.makedirs(my_folder)
        print(f"\tCreated folder: {my_folder}")
    else:
        print(f'\t{my_folder} is existing')
    return my_folder

## ------------------- Custom Tools --------------------------
def _determine_encoding(dataFile):
    import chardet

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

## ------------------------------------------------------------------
def _cleanUpSmiles(smi):
    ## text processing
    if "|" in smi:
        smi = smi.split("|")[0]
    smi = smi.replace("\n", "").replace("\r", "").replace("\r\n", "")

    ## rdkit smiles vadality checking
    try:
        mol = Chem.MolFromSmiles(smi)
        smi_rdkit = Chem.MolToSmiles(mol)
    except:
        smi_rdkit = np.nan
    return smi_rdkit

## ------------------------------------------------------------------
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
########################################################################
############################# Step-0. load argparses ###########################
########################################################################
def Step_0_load_args():
    print(f"==> Step 0: load the parameters ... ")
    ## 
    parser = argparse.ArgumentParser(description='This is the script to identify the MMPs from existing tables')
    ## input
    parser.add_argument('-q', action="store", type=int, default=None, help='D360 Query ID')
    parser.add_argument('-i', action="store", default=None, help='The input csv file for identify the MMPs')
    parser.add_argument('-d', action="store", default=',', help='The delimiter of input csv file for separate columns')

    ## data cols in the input
    parser.add_argument('--colName_cid', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colName_smi', action="store", default='Structure', help='The column name of SMILES')
    parser.add_argument('--colName_eid', action="store", default="External ID", help='The column name of external ID')
    parser.add_argument('--colName_prj', action="store", default="Concat;Project", help='The column name of Projects')

    parser.add_argument('--prop_dict_file', action="store", default="prop_cols_matches.json", help='The json file which specify the property of interest and the columns infomation')

    parser.add_argument('-o', '--output', action="store", default="MMPs_results", help='The name of output csv file to save the MMPs data')
    parser.add_argument('--tmpFolder', action="store", default='./tmp', help='The tmp folder')

    ## parse the arguments
    args = parser.parse_args()

    return args

##############################################################################################
############################ Step-1. data collection & clean ############################
##############################################################################################
def Step_1_load_data(my_query_id=3539, fileName_in=None, tmp_folder="./tmp", sep=','):
    ## count time
    beginTime = time.time()
    ## ------------------------------------------------------------------
    assert my_query_id is not None or fileName_in is not None, f"\tError, both <my_query_id> and <dataFile> are None"
    if my_query_id is not None:
        print(f"\tRun D360 query on ID {my_query_id}")
        ## download data from D360 using API
        dataTableFileName = dataDownload(my_query_id=my_query_id)
        print(f'\tAll data have been downloaded in file {dataTableFileName}')

        ## move the csv file to tmp folder
        fileName_in = f"{tmp_folder}/{dataTableFileName}"
        shutil.move(dataTableFileName, fileName_in)
        print(f"\tMove the downloaded file {dataTableFileName} to {fileName_in}")
    else:
        print(f"\tDirectly loading data from {fileName_in}")
        assert os.path.exists(fileName_in), f"File {fileName_in} does not exist"

    try:
        ## determine encoding type
        encoding = _determine_encoding(fileName_in)
        # encoding = 'ISO-8859-1'
        ## read csv file
        print(f"\tNow reading csv data using <{encoding}> encoding from {fileName_in}")
        dataTable = pd.read_csv(fileName_in, sep=sep, encoding=encoding).reset_index(drop=True)
    except Exception as e:
        print(f'\tError: cannot read output file {fileName_in}; error msg: {e}')
        dataTable = None
    else:
        print(f"\tThe loaded raw data has <{dataTable.shape[0]}> rows and {dataTable.shape[1]} columns")

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 1 <Loading csv data> complete, costs time = %ds ................\n" % (costTime))

    return dataTable

################################################################################################
###################### Step-2. clean up data and calculate property ############################
################################################################################################

def CheckThePropertyDataStats(dataTable, col_prop_prefix, propName):
    col_mod, col_num = f"{col_prop_prefix}(Mod)", f"{col_prop_prefix}(Num)"
    if (col_mod in dataTable) and (col_num in dataTable):
        cond_1 = (dataTable[col_mod]=='=')
        cond_2 = (dataTable[col_num].notna())
        # print(dataTable[cond_1].shape, dataTable[cond_2].shape)
        data_size_available = dataTable[cond_1 & cond_2].shape[0]
        print(f"\tThere are total <{data_size_available}> existing data for <{propName}>")
        passCheck = True
    else:
        print(f"\tWarning! The column <{col_prop_prefix}(Mod)/(Num)> is not in the table.")
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
            # print(f"\t------>change from {row[col_perctgF]} to np.nan, {row[col_vehicle]}")
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
        pctgF_PO, Clobs_IV = row[colName_pctF], row[colName_Clobs]
    except Exception as e:
        # print(f"\tWarning! Cannot get data for this row from column <{colName_pctgF}> or <{colName_Clobs}>")
        result = np.nan
    else:
        result = calc_EstFa_fromAdm(pctgF_PO, Clobs_IV, Species=Species)
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
            result = calc_mean(hERG_eIC50_list)
        else:
            result = np.nan
            # print(f"\tNo data in this row for column <{col_hERG_cmts}>")
    else:
        result = np.nan
        print(f"\tColumn <{col_hERG_cmts}> is not in the Table")
    return result

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
    print(f"2. Cleaning data ...")
    ## ------------------------------------------------------------------
    print(f'\tChecking the vadality of the SMILES using RDKit ...')
    dataTable[f"{colName_smi}_raw"] = dataTable[colName_smi].apply(lambda x: x)
    dataTable[colName_smi] = dataTable[colName_smi].apply(_cleanUpSmiles)

    ## ------------------------- remove invalid smiles -------------------------
    dataTable = dataTable.dropna(subset=[colName_mid, colName_smi]).reset_index(drop=True)
    print(f'\tThere are total <{dataTable.shape[0]}> molecules with valid SMILES<{colName_smi}>')

    ## ------------------------------------------------------------------
    for prop in dict_prop_cols:
        passCheck = CheckThePropertyDataStats(dataTable, col_prop_prefix=dict_prop_cols[prop], propName=prop)
        if passCheck:
            dataTable[prop] = dataTable.apply(lambda row: clean_up_prop_data(row, col_prop_prefix=dict_prop_cols[prop], propName=prop), axis=1)

        ## remove the 'elacridar' records
        if prop == 'Bioavailability':
            print(f"\t    The num rows with cleaned <{prop}> data (raw) is:", str(dataTable[dataTable[prop].notna()].shape[0]))
            dataTable[prop] = dataTable.apply(lambda row: rm_elacridar_records(row, col_perctgF=prop, col_vehicle='ADME PK;Concat;Vehicle'), axis=1)
            print(f"\t    The num rows with cleaned <{prop}> data (no elacridar) is:", str(dataTable[dataTable[prop].notna()].shape[0]))

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
        print(f"\t    The num rows with cleaned <{prop}> data is:", str(dataTable[dataTable[prop].notna()].shape[0]))

    ## ------------------------------------------------------------------
    colNames_basic = [colName_mid, colName_smi]
    colName_props = list(dict_prop_cols.keys())
    dataTable_4_mmp = dataTable[colNames_basic + colName_props]

    dateToday = datetime.datetime.today().strftime('%Y%b%d')
    dataTable_4_mmp.to_csv(f'{tmp_folder}/Data_4_MMP_{dateToday}.csv', index=False)
    print(f'\tThe prepared clean dataTable 4 FindMMPs have data shape {dataTable_4_mmp.shape}')

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 2 <data clean> complete, costs time = %ds ................\n" % (costTime))
    ## ------------------------------------------------------------------
    return dataTable_4_mmp

##############################################################################################
################################## Step-3 Run mmpdb analysis ######################################
##############################################################################################
## ---------------- prepare .smi and csv for MMPs analysis ----------------
def _prep_smi_file(dataTable, colName_prop_list, colName_mid='Compound Name', colName_smi='Structure', output_folder='./results'):
    ## count time
    print(f"\tNow preparing SMILES file and property CSV file for MMPs-DB analysis...")
    ## ------------------------------------------------------------------
    ## the SMILES file for fragmentation
    file_smi = f'{output_folder}/Compounds_All.smi'
    file_prop_csv = f'{output_folder}/Property_All.csv'
    delimiter_smi, delimiter_csv = ' ',  f"\t"
    ##
    data_dict_prop, data_dict_molID = {}, {}
    ## ----------------- write into .smi file directly -----------------
    with open(file_smi, "w") as output_file:
        # output_file.write(f'SMILES{delimiter_smi}ID' + "\n")
        for idx in dataTable.index:
            if idx % 1000 == 0:
                print(f"\t\trow {idx}")

            ## prepare the SMILES output (use row index to avoid the strange mol id from custom csv file)
            mol_id = str(idx)    #  mol_id = dataTable[colName_mid][idx]    
            mol_smi = dataTable[colName_smi][idx]

            ## prepare the SMILES output
            this_line = f'{mol_smi}{delimiter_smi}{mol_id}'
            output_file.write(this_line + "\n")  # Add a newline character after each string

            ## ----------------- prepare the property CSV  as dict -----------------
            data_dict_prop[idx] = {}
            data_dict_prop[idx]['ID'] = mol_id
            # data_dict_prop[idx]['idx_yjing'] = mol_id

            for prop_name in colName_prop_list:
                try:
                    if dataTable[prop_name].notna()[idx]:
                        mol_prop = float(dataTable[prop_name][idx])
                    else:
                        mol_prop = "*"
                except Exception as e:
                    data_dict_prop[idx][prop_name] = "*"
                    # print(f'\t---->Warning! This mol {mol_id} does not have a proper property value: {e}')
                else:
                    data_dict_prop[idx][prop_name] = mol_prop
            ## --------------------------------------------------------------------
            ## prep molID dict for query molID using index in the future step
            data_dict_molID[str(idx)] = dataTable[colName_mid][idx]    #.replace("\n", ";").strip()
        print(f'\tThe SMILES strings have been saved into .smi file: {file_smi}')
        
    ## ----------------- save the csv results -----------------
    data_table_prop = pd.DataFrame.from_dict(data_dict_prop).T
    data_table_prop.to_csv(file_prop_csv, index=False, sep=delimiter_csv)
    print(f'\tThe property data ({data_table_prop.shape}) have been saved into .csv file: {file_prop_csv}')
    # data_table_prop.head(3)

    return file_smi, file_prop_csv, data_dict_molID

## ---------------- basic cmd run ----------------
def _run_cmd(commandLine):
    # beginTime = time.time()

    # Use subprocess to execute the command
    process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # costTime = time.time()-beginTime
    # print(f"\tThis command costs time = %ds ................" % (costTime))
    return (output, error)

################################################################################################
def Step_3_mmp_analysis(dataTable, dict_prop_cols, colName_mid='Compound Name', colName_smi='Structure', output_folder='./results', symmetric=False):
    ## count time
    beginTime = time.time()
    print(f"3. Preparing SMILES file and property CSV file for MMPs-DB analysis...")

    ## ----------- prepare the Smiles file and property file -----------
    colName_prop_list = list(dict_prop_cols)
    file_smi, file_prop_csv, data_dict_molID = _prep_smi_file(dataTable, colName_prop_list, colName_mid, colName_smi, output_folder)

    ## -------------------- fragmentation SMILES -----------------------------
    file_fragdb = f'{output_folder}/Compounds_All.fragdb'
    commandLine_1 = ['mmpdb', 'fragment', file_smi, '-o', file_fragdb]
    (output_1, error_1) = _run_cmd(commandLine_1)
    print(f'\tThe fragmentation is completed and saved into file {file_fragdb}')

    ## ----------- Indexing to find the MMPs and load the activity data -----------
    file_mmpdb = f'{output_folder}/Compounds_All.mmpdb'
    commandLine_2 = ['mmpdb', 'index', file_fragdb, '-o', file_mmpdb, '--properties', file_prop_csv]

    if symmetric:
        commandLine_2.append('--symmetric')

    (output_2, error_2) = _run_cmd(commandLine_2)
    print(f'\tThe indexing/mmp generation is completed and saved into file {file_mmpdb}')

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 3 <MMPDB analysis> complete, costs time = %ds ................\n" % (costTime))
    return file_mmpdb, data_dict_molID

##############################################################################################
############################### Loading data from database ###################################
##############################################################################################
def _call_my_query(db_file, my_query):
    import sqlite3
    
    ## connect to the SQLIte database
    my_connection = sqlite3.connect(db_file)

    ## create a cursor object
    my_cursor = my_connection.cursor()

    ## excute the query
    my_cursor.execute(my_query)

    ## fetch all the rows
    rows = my_cursor.fetchall()
    
    # ## export the results
    # data_list = [row for row in rows]
    my_connection.close()
    return rows

## ------------- extract table data from SQLite DB ------------- 
def _extract_tables(db_file, table_name):
    ## get header info
    my_query_colName = f"PRAGMA table_info({table_name})"
    column_details = _call_my_query(db_file, my_query_colName)
    colName_list = [column[1] for column in column_details]

    ## get data info
    my_query_data = f"SELECT * FROM {table_name}"
    data_rows = _call_my_query(db_file, my_query_data)
    
    return colName_list, data_rows

def _write_2_csv(colName_list, data_rows, csv_file_name, delimiter=','):
    import csv
    with open(csv_file_name, 'w', newline='') as csvfh:
        writer = csv.writer(csvfh)    # , delimiter=delimiter
        ## --------- Write header ---------
        writer.writerow(colName_list)

        ## --------- Write data ---------
        print(f"\tNow start writing the data into csv")
        for i in range(0, len(data_rows)):
            writer.writerow(list(data_rows[i]))
            if i % 10**6 == 0:
                print(f"\t\trow-{i}")
    print(f"\tNow the table data were saved into <{csv_file_name}>")
    return None

################################################################################################
def Step_4_extract_data_from_DB(file_mmpdb, tmp_folder):
    ## count time
    beginTime = time.time()
    print(f"4. Now extracting tables from MMPs database ...")
    ## ------------------------------------------------------------------
    dataDict_csvFiles = {}
    for table_name in ["pair", "compound", "compound_property", "property_name", "constant_smiles",
                    "rule", "rule_smiles", "rule_environment", "rule_environment_statistics", "environment_fingerprint"]:
        
        print(f"\tNow processing the table <{table_name}>")
        colName_list, data_rows = _extract_tables(file_mmpdb, table_name)       

        ## --------- write output ---------
        ## define folder and csv fileName
        subFolderDB = _folderChecker(f"{tmp_folder}/DB_tables")
        table_out = f"{subFolderDB}/DB_table_{table_name}.csv"
        ## write 2 csv
        _write_2_csv(colName_list, data_rows, table_out)

        print(f"\t<{table_name}> table has been saved into {table_out}\n")
        dataDict_csvFiles[table_name] = table_out
        # print(table_name)

    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 4 <Extracting data from MMPs DB> complete, costs time = %ds ................\n" % (costTime))    
    return dataDict_csvFiles

##############################################################################################
############################### cleanning the data from database #############################
##############################################################################################
## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------

################################################################################################
def Step_5_MMPs_DataClean(dataDict_tables, add_symetric=True):
    ## count time
    beginTime = time.time()
    print(f"5. Now clean up the MMPs data ...")

    ## ------------------------------------------------------------------
    ## ------------- build the dataDict of pairs -------------
    print(f"\tNow start cleanning up the dataDict of pairs ...\n")
    dataDict = {}
    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 5 <Final data clean> complete, costs time = %ds ................\n" % (costTime))
    return dataDict

##############################################################################################
#################################### exporting the data ######################################
##############################################################################################
def Step_6_MMPs_dataClean(dataDict, colName_prop_list, data_dict_molID, fileName_out):
    ## count time
    beginTime = time.time()
    print(f"6. Now export a table of pairs ...")
    ## ------------------------------------------------------------------
    dataTable = pd.DataFrame.from_dict(dataDict).T
    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 6 <mmp data export> complete, output saved to {fileName_out}, costs time = %ds ................\n" % (costTime))
    return dataTable

##############################################################################################
###################################### main function #########################################
##############################################################################################
def main():
    ## =================== get parameters from arguments ==================== ##
    args = Step_0_load_args()

    if True:
        ## ----------- input -----------
        my_query_id = args.q    # 3539
        fileName_in = args.i
        sep = args.d

        ## ----------- input cols -----------
        colName_mid = args.colName_cid    # 'Compound Name'
        colName_smi = args.colName_smi    # 'Structure' or 'Smiles'
        # colName_eid = args.colName_eid    # 'Concat;External Id'
        # colName_proj = args.colName_prj    # 'Concat;Project'

        ## ----------- props -----------
        # ## Reading JSON data from a file
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
        ## ----------- output -----------

        ## ----------- create folders -----------
        folderName_tmp = f"./tmp"
        tmp_folder = _folderChecker(folderName_tmp)
        output_folder = _folderChecker(f"./results")
    
    ## ============================ run the code ============================
    #### Step-1. download & load data from D360
    dataTable_raw = Step_1_load_data(my_query_id, fileName_in, tmp_folder)

    ## ------------------------------------------------------------------
    #### Step-2. clean up dataTable
    dataTable_4_mmp = Step_2_clean_data(dataTable_raw, dict_prop_cols, colName_mid, colName_smi, tmp_folder)

    ## ------------------------------------------------------------------
    #### Step-3. MMPs analysis
    file_mmpdb, data_dict_molID = Step_3_mmp_analysis(dataTable_4_mmp, dict_prop_cols, colName_mid, colName_smi, output_folder, symmetric=True)

    ## ------------------------------------------------------------------
    #### Step-4. Extracing all tables from database file
    ## ------------------------------------------------------------------
    #### Step-5. Clean up the MMPs data from the DB

    ## ------------------------------------------------------------------
    #### Step-6. save the results

if __name__ == '__main__':
    main()