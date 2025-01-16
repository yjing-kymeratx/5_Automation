#!/fsx/home/yjing/apps/anaconda3/env/yjing/bin python
# findMMPs_new
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
from rdkit.Chem import rdFMCS, rdMolDescriptors, Descriptors

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
def _calc_mw_rdkit(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
    except Exception as e:
        mw = np.nan
    return mw

########################################################################
############################# Step-0. load argparses ###########################
########################################################################
def Step_0_load_args():
    print(f"==> Step 0: load the parameters ... ")
    ## 
    parser = argparse.ArgumentParser(description='This is the script to identify the MMPs from existing tables')
    ## input
    # parser.add_argument('-q', action="store", type=int, default=None, help='D360 Query ID')
    parser.add_argument('-i', action="store", default=None, help='The input csv file for identify the MMPs')
    parser.add_argument('-d', action="store", default=',', help='The delimiter of input csv file for separate columns')

    ## data cols in the input
    parser.add_argument('--colName_cid', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colName_smi', action="store", default='Structure', help='The column name of SMILES')
    parser.add_argument('--colName_eid', action="store", default="External ID", help='The column name of external ID')
    parser.add_argument('--colName_prj', action="store", default="Concat;Project", help='The column name of Projects')

    # parser.add_argument('--prop_dict_file', action="store", default="prop_cols_matches.json", help='The json file which specify the property of interest and the columns infomation')
    parser.add_argument('--colName_assay', action="store", default='ActivityColumn1,ActivityColumn2', help='The column names of the assay values, separated by comma with no space')

    parser.add_argument('-o', '--output', action="store", default="MMPs_results", help='The name of output csv file to save the MMPs data')
    parser.add_argument('--tmpFolder', action="store", default='./tmp', help='The tmp folder')

    ## parse the arguments
    args = parser.parse_args()

    return args


##############################################################################################
############################ Step-1. data collection & clean ############################
##############################################################################################
def Step_1_load_data(fileName_in, colName_mid, colName_smi, colNames_activity, sep=','):
    ## count time
    beginTime = time.time()
    ## ------------------------------------------------------------------
    print(f"1. Loading csv from {fileName_in}")
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

    ## ------------------------- check the columns -------------------------
    assert colName_mid in dataTable.columns, f"Error! The mol ID column <{colName_mid}> does not exist"
    print(f"\tColumn for compound ID is {colName_mid}")
    
    assert colName_smi in dataTable.columns, f"Error! The mol SMILES column <{colName_smi}> does not exist"
    print(f"\tColumn for compound SMILES is {colName_smi}")

    ## ------------------------- check the prop columns -------------------------
    if colNames_activity == "*":
        if colNames_activity not in dataTable.columns:
            colName_prop_list = [col for col in dataTable.columns if col not in [colName_mid, colName_smi]]
        else:
            print(f"\tError! You decide to use all columns but there is a column named * in the dataset")
            colName_prop_list = [colNames_activity]
        print(f"\tUsing columns {colName_prop_list}")
    else:
        colName_prop_list = colNames_activity.split(',')
        print(f"\tColumns for compound activity includes {colName_prop_list}")
        for prop_name in colName_prop_list:
            if prop_name not in dataTable.columns:
                print(f"\t---->Warning! {prop_name} is not in the csv file, pls check ...")
                colName_prop_list.remove(prop_name)
    ## add molecular weight for general pair analysis
    if 'MW' not in colName_prop_list:
        colName_prop_list.append('MW')
    ## ------------------------------------------------------------------
    costTime = time.time()-beginTime
    print(f"==> Step 1 <Loading csv data> complete, costs time = %ds ................\n" % (costTime))

    return dataTable, colName_prop_list

################################################################################################
###################### Step-2. clean up data and calculate property ############################
################################################################################################
################################################################################################
def Step_2_clean_data(dataTable, colName_prop_list, colName_mid, colName_smi, tmp_folder="./tmp"):
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

    ## ------------------------- add molecular weight ------------------
    dataTable_4_mmp['MW'] = dataTable_4_mmp[colName_smi].apply(_calc_mw_rdkit)

    ## ------------------------------------------------------------------
    colNames_basic = [colName_mid, colName_smi, 'MW']
    colName_props = colName_prop_list
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
def Step_3_mmp_analysis(dataTable, colName_prop_list, colName_mid='Compound Name', colName_smi='Structure', output_folder='./results', symmetric=False):
    ## count time
    beginTime = time.time()
    print(f"3. Preparing SMILES file and property CSV file for MMPs-DB analysis...")

    ## ----------- prepare the Smiles file and property file -----------
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
def _findPropValue(dbTable_propValue, cid, prop_id, average=False):
    cond_1 = (dbTable_propValue["compound_id"]==cid)
    cond_2 = (dbTable_propValue["property_name_id"]==prop_id)
    
    match_data = dbTable_propValue[cond_1 & cond_2]["value"].values

    if match_data.shape[0] <= 0:
        result = np.nan
    else:
        if average:
            if match_data.shape[0] > 1:
                print(f"\t\tWarning! Compound {cid} has multiple <{prop_id}> values\n")
                result = np.meam(match_data)
            else:
                result = match_data[0]
        else:
            result = np.array2string(match_data, separator=',')
    return result

## ------------------------------------------------------------------------------------------------------
def __SelectTheConstant(pair_detail, dbTable_constSmi, sele_rule="max"):
    const_id_sele = list(pair_detail.keys())[0]
    const_mw_sele, const_smi_sele = -1, np.nan
    sele_rule = "max"    ## ["max", "min"]

    for const_id in pair_detail:
        try:
            const_smi = dbTable_constSmi[dbTable_constSmi['id']==const_id]['smiles'].values[0]
            const_mol = Chem.MolFromSmiles(const_smi)
            const_mw = Descriptors.MolWt(const_mol)
        except Exception as e:
            print(f"\tError when getting the MW of const id <{const_id}>; Error msgL {e}")
            const_mw = np.nan
        else:
            if sele_rule == "max":
                if const_mw_sele < const_mw:
                    const_id_sele, const_mw_sele, const_smi_sele = const_id, const_mw, const_smi
            elif sele_rule == "min":
                if const_mw_sele > 0 and const_mw_sele > const_mw:
                    const_id_sele, const_mw_sele, const_smi_sele = const_id, const_mw, const_smi
            else:
                print(f"<sele_rule> should be either max or min")
    return const_id_sele, const_smi_sele

## ------------------------------------------------------------------------------------------------------
def __SelectTheRuleEnv(rule_env_id_list, dbTable_rule, dbTable_ruleSmi, dbTable_ruleEnv, radius=0):
    from_smiles, to_smiles = np.nan, np.nan

    dbTable_ruleEnv_rs = dbTable_ruleEnv[dbTable_ruleEnv['radius']==radius]
    for rule_env_id in rule_env_id_list:
        ## find rule env
        if rule_env_id not in dbTable_ruleEnv_rs['id'].to_list():
            continue
        row_rule_env = dbTable_ruleEnv_rs[dbTable_ruleEnv_rs['id']==rule_env_id]
        
        ## find rule
        rule_id = row_rule_env['rule_id'].values[0]
        if rule_id not in dbTable_rule['id'].to_list():
            continue
        row_rule = dbTable_rule[dbTable_rule['id']==rule_id]

        ## find from smi
        from_smiles_id = row_rule['from_smiles_id'].values[0]
        row_from_smi = dbTable_ruleSmi[dbTable_ruleSmi['id']==from_smiles_id]
        from_smiles = row_from_smi['smiles'].values[0]
        # from_smiles_nh = row_from_smi['num_heavies'].values[0]

        ## find to smi
        to_smiles_id = row_rule['to_smiles_id'].values[0]    
        row_to_smi = dbTable_ruleSmi[dbTable_ruleSmi['id']==to_smiles_id]
        to_smiles = row_to_smi['smiles'].values[0]
        # to_smiles_nh = row_to_smi['num_heavies'].values[0]

        break
    return from_smiles, to_smiles

## ------------------------------------------------------------------------------------------------------
def _findTranSmi(pair_detail, dataDict_tables, sele_rule="max", radius=0):
    ## get the individual database Tables
    dbTable_constSmi = dataDict_tables["constant_smiles"]
    dbTable_ruleEnv = dataDict_tables["rule_environment"]
    dbTable_rule = dataDict_tables["rule"]
    dbTable_ruleSmi = dataDict_tables["rule_smiles"]

    const_id_sele, const_smi_sele = __SelectTheConstant(pair_detail, dbTable_constSmi, sele_rule=sele_rule)
    rule_env_id_list = pair_detail[const_id_sele]
    # radius = 0    # [0, 1, 2, 3, 4, 5]
    from_smiles, to_smiles = __SelectTheRuleEnv(rule_env_id_list, dbTable_rule, dbTable_ruleSmi, dbTable_ruleEnv, radius=radius)
    return const_smi_sele, from_smiles, to_smiles

################################################################################################
def Step_5_MMPs_DataClean(dataDict_tables, add_symetric=True):
    ## count time
    beginTime = time.time()
    print(f"5. Now clean up the MMPs data ...")
    
    ## ------------------------------------------------------------------
    ## get the individual database Tables
    dataTable_pair = dataDict_tables["pair"]
    dbTable_cmpd = dataDict_tables["compound"]
    dbTable_propName = dataDict_tables["property_name"]
    dbTable_propValue = dataDict_tables["compound_property"]

    ## ------------- build the dataDict of pairs -------------
    print(f"\tNow start cleanning up the dataDict of pairs ...\n")
    dataDict = {}
    for idx in dataTable_pair.index:
        pair_idx = dataTable_pair['id'][idx]
        cid_1 = int(dataTable_pair['compound1_id'][idx])
        cid_2 = int(dataTable_pair['compound2_id'][idx])
        const_id = dataTable_pair['constant_id'][idx]
        rule_env_id = dataTable_pair['rule_environment_id'][idx]

        ## initialize the sub-dict
        pair_info = f"{cid_1}==>{cid_2}"
        try:
            pair_list = sorted([cid_1, cid_2], reverse=False)
        except Exception as e:
            pass    
        
        if pair_info not in dataDict:
            ## add pair basic info
            dataDict[pair_info] = {}
            dataDict[pair_info]["pair_info"] = pair_info
            dataDict[pair_info]["pair_id"] = f"({min([cid_1, cid_2])},{max([cid_1, cid_2])})"
            dataDict[pair_info]["compound1_id"] = cid_1
            dataDict[pair_info]["compound2_id"] = cid_2
            dataDict[pair_info]["pair_detail"] = {}

            ## add compound info
            dataDict[pair_info]["From_mol_id"] = dbTable_cmpd['public_id'][cid_1]
            dataDict[pair_info]["To_mol_id"] = dbTable_cmpd['public_id'][cid_2]
            smi_1, smi_2 = dbTable_cmpd['input_smiles'][cid_1], dbTable_cmpd['input_smiles'][cid_2]
            dataDict[pair_info]["From_Structure"] = smi_1
            dataDict[pair_info]["To_Structure"] = smi_2
            
            ## add shared structure
            # dataDict[pair_info]["SharedSubstructure"] = fun_tbd(smi_1, smi_2)

            ## add compound prop info
            for prop_id in dbTable_propName.index:
                prop_name = dbTable_propName['name'][prop_id]
                dataDict[pair_info][f"From_{prop_name}"] = _findPropValue(dbTable_propValue, cid_1, prop_id, average=True)
                dataDict[pair_info][f"To_{prop_name}"] = _findPropValue(dbTable_propValue, cid_2, prop_id, average=True)
                ## add delta value change
                try:
                    delta_value = dataDict[pair_info][f"To_{prop_name}"] - dataDict[pair_info][f"From_{prop_name}"]
                except Exception as e:
                    delta_value = np.nan
                    
                dataDict[pair_info][f"Delta_{prop_name}"] = delta_value

        ## add pair details information (constant part)
        if const_id not in dataDict[pair_info]["pair_detail"]:
            dataDict[pair_info]["pair_detail"][const_id] = []
        
        ## add pair details information (rule_env)
        if rule_env_id not in dataDict[pair_info]["pair_detail"][const_id]:
            dataDict[pair_info]["pair_detail"][const_id].append(rule_env_id)
    print(f"\t\tOriginal num_pairs in dataDict: {len(dataDict)}\n")
    
    ## ------------- add the symetric pairs if not exist -------------
    tran_smi = True
    radius = 0    # [0, 1, 2, 3, 4, 5]
    sele_rule = "max"    # ["max", "min"]
    if tran_smi:
        for pair_info in dataDict:
            pair_detail = dataDict[pair_info]["pair_detail"]
            const_smi_sele, from_smiles, to_smiles = _findTranSmi(pair_detail, dataDict_tables, sele_rule=sele_rule, radius=radius)
            dataDict[pair_info]["constant_smiles"] = const_smi_sele
            dataDict[pair_info]["from_smiles"] = from_smiles
            dataDict[pair_info]["to_smiles"] = to_smiles
        
    ## ------------- add the symetric pairs if not exist -------------
    if add_symetric:
        print(f"\t\tNow adding symetric pairs ...")
        list_pair_info_4loop = copy.deepcopy(list(dataDict.keys()))
        list_pair_info_4check = copy.deepcopy(list(dataDict.keys()))
        for pair_info in list_pair_info_4loop:
            if pair_info in list_pair_info_4check:
                list_pair_info_4check.remove(pair_info)

                ## reverse pair
                cid_1, cid_2 = pair_info.split("==>")
                pair_info_revs = f"{cid_2}==>{cid_1}"
                if pair_info_revs in list_pair_info_4check:
                    list_pair_info_4check.remove(pair_info_revs)
                else:
                    ## if reversed pair not in check list, add it in the dict
                    dataDict[pair_info_revs] = {}
                    dataDict[pair_info_revs]["pair_info"] = pair_info_revs
                    dataDict[pair_info_revs]["pair_id"] = dataDict[pair_info]["pair_id"]
                    dataDict[pair_info_revs]["constant_smiles"] = dataDict[pair_info]["constant_smiles"]
                    dataDict[pair_info_revs]["pair_detail"] = {key: [] for key in dataDict[pair_info]["pair_detail"]}

                    dataDict[pair_info_revs]["From_mol_id"] = dataDict[pair_info]["To_mol_id"]
                    dataDict[pair_info_revs]["To_mol_id"] = dataDict[pair_info]["From_mol_id"]
                    dataDict[pair_info_revs]["From_Structure"] = dataDict[pair_info]["To_Structure"]
                    dataDict[pair_info_revs]["from_smiles"] = dataDict[pair_info]["to_smiles"]
                    dataDict[pair_info_revs]["to_smiles"] = dataDict[pair_info]["from_smiles"]
                    
                    for tmp_key in dataDict[pair_info]:
                        if tmp_key[0:6] == 'Delta_':
                            try:
                                delta_symetric = dataDict[pair_info][tmp_key] * -1
                            except Exception as e:
                                delta_symetric = np.nan
                            dataDict[pair_info_revs][tmp_key] = delta_symetric

                        elif tmp_key[0:5] == 'From_':
                            tmp_key_reverse = 'To_' + tmp_key[5:]
                            dataDict[pair_info_revs][tmp_key_reverse] = dataDict[pair_info][tmp_key]
                        
                        elif tmp_key[0:3] == "To_":
                            tmp_key_reverse = 'From_' + tmp_key[3:]
                            dataDict[pair_info_revs][tmp_key_reverse] = dataDict[pair_info][tmp_key]
                        else:
                            pass
            else:
                ## this pair was removed from check list because it's the revs pair of another pair
                pass
        print(f"\t\tNew num_pairs in symetric dataDict: {len(dataDict)}")
    
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

    ## ----------- add the  -----------
    dataTable['Num_Counts'] = dataTable['pair_detail'].apply(lambda x: len(x))
    dataTable = dataTable.sort_values(by=["pair_id", "pair_info"], ascending=[True, True]).reset_index(drop=True)

    ## ----------- add the number of cuts -----------
    dataTable['Num_cuts'] = dataTable['constant_smiles'].apply(lambda x: len(x.split('.')))

    ## ----------- re-arrange the columns -----------
    col_bioassay = []
    for x in colName_prop_list:
        col_bioassay.append(f"From_{x}")
        col_bioassay.append(f"To_{x}")
        col_bioassay.append(f"Delta_{x}")

    col_basic = ['From_mol_id', 'To_mol_id', 'From_Structure', 'To_Structure', 
                 'from_smiles', 'to_smiles', 'constant_smiles', 'Num_cuts']
    col_pairinfo = ['pair_detail']    #, 'pair_info', 'pair_id', 'Num_Counts'
    dataTable = dataTable[col_basic + col_bioassay + col_pairinfo]
    # dataTable = dataTable.drop(columns=["mw_ying_from_value", "mw_ying_to_value"])
    # print(dataTable.columns)

    ## ----------- mapping mol id -----------
    dataTable['From_mol_id'] = dataTable['From_mol_id'].apply(lambda x: data_dict_molID[x] if x in data_dict_molID else x)
    dataTable['To_mol_id'] = dataTable['To_mol_id'].apply(lambda x: data_dict_molID[x] if x in data_dict_molID else x)

    ## ----------- exporting -----------
    dataTable.to_csv(f"{fileName_out}", index=False)

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
        # my_query_id = args.q    # 3539
        fileName_in = args.i
        sep = args.d

        ## ----------- input cols -----------
        colName_mid = args.colName_cid    # 'Compound Name'
        colName_smi = args.colName_smi    # 'Structure' or 'Smiles'
        # colName_eid = args.colName_eid    # 'Concat;External Id'
        # colName_proj = args.colName_prj    # 'Concat;Project'

        ## ----------- props -----------
        colNames_activity = args.colName_assay
        # ## Reading JSON data from a file
        # prop_dict_file = args.prop_dict_file
        # print(prop_dict_file)
        # with open(prop_dict_file, 'r') as infile:
        #     dict_prop_cols = json.load(infile)        
        ## ----------- output -----------
        fileName_out = str(args.output)
        if fileName_out[-4:] != '.csv':
            fileName_out = fileName_out + ".csv"

        ## ----------- create folders -----------
        folderName_tmp = args.tmpFolder
        tmp_folder = _folderChecker(folderName_tmp)
        output_folder = _folderChecker(f"./results")

    ## ============================ run the code ============================
    #### Step-1. Load the raw data from csv file
    dataTable_raw, colName_prop_list = Step_1_load_data(fileName_in, colName_mid, colName_smi, colNames_activity, sep=sep)

    ## ------------------------------------------------------------------
    #### Step-2. clean up dataTable
    dataTable_4mmp = Step_2_clean_data(dataTable_raw, colName_prop_list, colName_mid, colName_smi, tmp_folder="./tmp")

    ## ------------------------------------------------------------------
    #### Step-3. MMPs analysis
    file_mmpdb, data_dict_molID = Step_3_mmp_analysis(dataTable_4mmp, colName_prop_list, colName_mid, colName_smi, output_folder, symmetric=True)

    ## ------------------------------------------------------------------
    #### Step-4. Extracing all tables from database file
    dataDict_tables = Step_4_extract_data_from_DB(file_mmpdb, tmp_folder)

    ## ------------------------------------------------------------------
    #### Step-5. Clean up the MMPs data from the DB
    dataDict = Step_5_MMPs_DataClean(dataDict_tables, add_symetric=True)

    ## ------------------------------------------------------------------
    #### Step-6. save the results
    dataTable_out = Step_6_MMPs_dataClean(dataDict, colName_prop_list, data_dict_molID, fileName_out)

if __name__ == '__main__':
    main()


