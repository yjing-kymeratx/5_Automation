#!/fsx/home/yjing/apps/anaconda3/env/yjing/bin python

##############################################################################################
##################################### load packages ###########################################
##############################################################################################
import os
import time
import copy
import pickle
import chardet
import sqlite3
import argparse
import datetime
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem

##############################################################################################
##################################### Custom Tools ###########################################
##############################################################################################
def _determine_encoding(dataFile):
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

def _defineTmpFolder(folderName_tmp=None):
    if folderName_tmp is None:
        folderName_tmp = 'Tmp'
    
    folderName_tmp = os.path.join(os.getcwd(), folderName_tmp)
    os.makedirs(folderName_tmp) if not os.path.exists(folderName_tmp) else print(f'\t---->{folderName_tmp} is existing')
    return folderName_tmp

##############################################################################################
########################### Load original csv for MMPs analysis ##############################
##############################################################################################
def CSV_loader(fileName_in, colName_mid, colName_smi, colNames_activity, sep=','):
    print(f"1. Loading csv from {fileName_in}")
    assert os.path.exists(fileName_in), f"File {fileName_in} does not exist"
    ##
    encoding = _determine_encoding(fileName_in)
    ##
    dataTable_raw = pd.read_csv(fileName_in, sep=sep, encoding=encoding)
    print(f"\tThe original csv file has {dataTable_raw.shape[0]} rows and {dataTable_raw.shape[1]} columns")
    print(f"\tColumn for compound ID is {colName_mid}")
    print(f"\tColumn for compound SMILES is {colName_smi}")

    colName_prop_list = colNames_activity.split(',')
    print(f"\tColumns for compound activity includes {colName_prop_list}")
    for prop_name in colName_prop_list:
        if prop_name not in dataTable_raw.columns:
            print(f"\t---->Warning! {prop_name} is not in the csv file, pls check ...")
            colName_prop_list.remove(prop_name)
    ##
    dataTable_raw = dataTable_raw.dropna(subset=[colName_mid, colName_smi]).reset_index(drop=True)
    print(f"\tThere are total {dataTable_raw.shape[0]} molecules in the csv with Structure(SMILES)")

    return dataTable_raw, colName_prop_list

##############################################################################################
######################## Prepare .smi and csv for MMPs analysis ##############################
##############################################################################################
def Smiles_Prep(dataTable_raw, colName_mid, colName_smi, colName_prop_list, folderName_tmp):
    print(f"2. Prepare the smi & prop file for MMPs-DB analysis")
    ## the SMILES file for fragmentation
    file_smi = f'{folderName_tmp}/Compounds_All.smi'
    file_prop_csv = f'{folderName_tmp}/Property_All.csv'
    delimiter = ' '

    data_dict_prop = {}
    with open(file_smi, "w") as output_file:
        # output_file.write(f'SMILES{delimiter}ID' + "\n")
        for idx in dataTable_raw.index:
            mol_id = dataTable_raw[colName_mid][idx]
            mol_smi = dataTable_raw[colName_smi][idx]

            ## prepare the SMILES output
            this_line = f'{mol_smi}{delimiter}{mol_id}'
            output_file.write(this_line + "\n")  # Add a newline character after each string

            ## prepare the property CSV output
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
                    print(f'\t---->Warning! This mol {mol_id} does not have a proper property value: {e}')
                else:
                    data_dict_prop[idx][prop_name] = mol_prop
        print(f'\tThe SMILES strings have been saved into .smi file: {file_smi}')
        
    ## save the csv results
    data_table_prop = pd.DataFrame.from_dict(data_dict_prop).T
    data_table_prop.to_csv(file_prop_csv, index=False, sep=delimiter)
    print(f'\tThe property data have been saved into .csv file: {file_smi}')
    return file_smi, file_prop_csv

##############################################################################################
##################################### Fragment the SMILES ####################################
##############################################################################################
def Smiles_fragmentation(folderName_tmp, file_smi):
    print(f"3. Fragment the SMILES")
    file_fragdb = f'{folderName_tmp}/Compounds_All.fragdb'
    commandLine = ['mmpdb', 'fragment', file_smi, '-o', file_fragdb]
    process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f'\tThe fragmentation is completed and saved into file {file_fragdb}')
    return file_fragdb

##############################################################################################
################## Indexing to find the MMPs and load the activity data ######################
##############################################################################################
def Index_LinkActivity(folderName_tmp, file_fragdb, file_prop_csv):
    print(f"4.1 Indexing to find the matched molecular pairs in the fragment file")
    print(f"4.2 Now load the activity/property data")
    file_mmpdb = f'{folderName_tmp}/Compounds_All.mmpdb'
    commandLine = ['mmpdb', 'index', file_fragdb, '-o', file_mmpdb, '--properties', file_prop_csv]
    process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f'\tThe indexing/mmp generation is completed and saved into file {file_mmpdb}')
    return file_mmpdb

##############################################################################################
############################### Loading data from database ###################################
##############################################################################################
def _call_my_query(db_file, my_query):
    ## connect to the SQLIte database
    my_connection = sqlite3.connect(db_file)

    ## create a cursor object
    my_cursor = my_connection.cursor()

    ## excute the query
    my_cursor.execute(my_query)

    ## fetch all the rows
    rows = my_cursor.fetchall()
    
    ## export the results
    data_list = [row for row in rows]

    my_connection.close()
    return data_list

def _extract_tables(db_file, table_name):
    ## extract table data from SQLite DB
    my_query_colName = f"PRAGMA table_info({table_name})"
    colName_list = _call_my_query(db_file, my_query_colName)

    my_query_data = f"SELECT * FROM {table_name}"
    data_list = _call_my_query(db_file, my_query_data)

    ## clean up data
    dataDict = {}
    for row_tuple in data_list:
        idx = row_tuple[0]
        dataDict[idx] = {}

        for col in colName_list:
            colIdx, colName = col[0], col[1]
            dataDict[idx][colName] = row_tuple[colIdx]
    return dataDict
    
def DataExtractionFromDB(file_mmpdb, folderName_tmp, savedict=False):
    print(f"5. Now extracting tables from MMPs database")
    dataDict_tables = {}
    for table_name in ["pair", "compound", "compound_property", "property_name", "constant_smiles",
                    "rule", "rule_smiles", "rule_environment", "rule_environment_statistics", "environment_fingerprint"]:
        dataDict_table = _extract_tables(file_mmpdb, table_name)
        dataTable_table = pd.DataFrame.from_dict(dataDict_table).T

        ## output
        subFolderDB = f"{folderName_tmp}/DB_tables"
        os.makedirs(subFolderDB) if not os.path.exists(subFolderDB) else print(f'\t---->{subFolderDB} is existing')
        table_out = f"{subFolderDB}/DB_table_{table_name}.csv"
        dataTable_table.to_csv(table_out, index=False)
        print(f"\t<{table_name}> table has been saved into {table_out}")
        dataDict_tables[table_name] = dataTable_table
        # print(table_name)
    
    if savedict:
        tableDict_out = f"{subFolderDB}/DB_table_all.dict"
        with open(tableDict_out, "wb") as ofh:
            pickle.dump(dataDict_tables, ofh)
            print(f"\tAll tables have been dumped into {tableDict_out}")
    return dataDict_tables

##############################################################################################
############################### Loading data from database ###################################
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
                print(f"\t\tWarning! Compound {cid} has multiple <{prop_id}> values")
                result = np.meam(match_data)
            else:
                result = match_data[0]
        else:
            result = np.array2string(match_data, separator=',')
    return result

def MMPs_DataClean(dataDict_tables, add_symetric=True):
    print(f"6. Now clean up the MMPs data and export a table of pairs")
    ## get the individual database Tables
    dataTable_pair = dataDict_tables["pair"]
    dbTable_cmpd = dataDict_tables["compound"]
    dbTable_propName = dataDict_tables["property_name"]
    dbTable_propValue = dataDict_tables["compound_property"]

    ## ------------- build the dataDict of pairs -------------
    print(f"\tNow start cleanning up the dataDict of pairs ...")
    dataDict = {}
    for idx in dataTable_pair.index:
        pair_idx = dataTable_pair['id'][idx]
        cid_1 = dataTable_pair['compound1_id'][idx]
        cid_2 = dataTable_pair['compound2_id'][idx]
        const_id = dataTable_pair['constant_id'][idx]
        rule_env_id = dataTable_pair['rule_environment_id'][idx]

        ## initialize the sub-dict
        pair_info = f"{cid_1}==>{cid_2}"
        if pair_info not in dataDict:
            ## add pair basic info
            dataDict[pair_info] = {}
            dataDict[pair_info]["pair_info"] = pair_info
            dataDict[pair_info]["pair_id"] = f"({min([cid_1, cid_2])},{max([cid_1, cid_2])})"
            dataDict[pair_info]["compound1_id"] = cid_1
            dataDict[pair_info]["compound2_id"] = cid_2
            dataDict[pair_info]["pair_detail"] = {}

            ## add compound info
            dataDict[pair_info]["KT_id_1"] = dbTable_cmpd['public_id'][cid_1]
            dataDict[pair_info]["KT_id_2"] = dbTable_cmpd['public_id'][cid_2]
            dataDict[pair_info]["Smiles_1"] = dbTable_cmpd['input_smiles'][cid_1]
            dataDict[pair_info]["Smiles_2"] = dbTable_cmpd['input_smiles'][cid_2]

            ## add compound prop info
            for prop_id in dbTable_propName.index:
                prop_name = dbTable_propName['name'][prop_id]

                dataDict[pair_info][f"{prop_name}_1"] = _findPropValue(dbTable_propValue, cid_1, prop_id, average=True)
                dataDict[pair_info][f"{prop_name}_2"] = _findPropValue(dbTable_propValue, cid_2, prop_id, average=True)

        ## add pair details information (constant part)
        if const_id not in dataDict[pair_info]["pair_detail"]:
            dataDict[pair_info]["pair_detail"][const_id] = []
        
        ## add pair details information (rule_env)
        if rule_env_id not in dataDict[pair_info]["pair_detail"][const_id]:
            dataDict[pair_info]["pair_detail"][const_id].append(rule_env_id)

    print(f"\t\tOriginal num_pairs in dataDict: {len(dataDict)}")
    
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
                    dataDict[pair_info_revs]["pair_detail"] = {key: [] for key in dataDict[pair_info]["pair_detail"]}

                    for tmp_key in dataDict[pair_info]:
                        if '1' in tmp_key:
                            dataDict[pair_info_revs][tmp_key] = dataDict[pair_info][tmp_key.replace('1', '2')]
                        elif '2' in tmp_key:
                            dataDict[pair_info_revs][tmp_key] = dataDict[pair_info][tmp_key.replace('2', '1')]
                        else:
                            pass
            else:
                ## this pair was removed from check list because it's the revs pair of another pair
                pass
        print(f"\t\tNew num_pairs in symetric dataDict: {len(dataDict)}")
    return dataDict

##############################################################################################
###################################### main function #########################################
##############################################################################################
def main():
    beginTime = time.time()

    ## ------------------ argument parser ------------------ ##
    parser = argparse.ArgumentParser(description='xxxxxxxxx')

    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file for identify the MMPs')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('-o', '--output', action="store", default="MMPs_results", help='The name of output csv file to save the MMPs data')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Smiles', help='The column name of the compound smiles')
    parser.add_argument('--colAssay', action="store", default='ActivityColumn1,ActivityColumn2', help='The column names of the assay values, separated by comma with no space')
    parser.add_argument('--tmpFolder', action="store", default='Tmp', help='The column names of the assay values, separated by comma with no space')


    args = parser.parse_args()    ## parse the arguments

    ## get parameters from arguments
    fileName_in = args.input    # './Data_ADMET_4_MMP_2024Aug27.csv' 
    sep = args.delimiter    # ','     
    fileName_out = f"MMPs_results.csv" if args.output is None else f"{args.output.replace(".csv", "")}.csv" 

    colName_mid = args.colId    # 'Compound Name'
    colName_smi = args.colSmi    # 'Smiles'
    colNames_activity = args.colAssay    # 'F%_Rat,EstFa_Rat,permeability,efflux,hERG_IC50,hERG_mixedIC50,logD_CDD'

    folderName_tmp = _defineTmpFolder(args.tmpFolder)     # 'Tmp'

    ## ------------------ run the code ------------------ ##
    ## 1. Load the raw data from csv file
    dataTable_raw, colName_prop_list = CSV_loader(fileName_in, colName_mid, colName_smi, colNames_activity, sep=',')
    time_1 = time.time()
    print(f"\t==> Loading csv data completed, costing time {int(time_1-beginTime)}s")

    ## 2. Prepare the SMILES file and property CSV file
    file_smi, file_prop_csv = Smiles_Prep(dataTable_raw, colName_mid, colName_smi, colName_prop_list, folderName_tmp)
    time_2 = time.time()
    print(f"\t==> Prepare the smi & prop file completed, costing time {int(time_2-time_1)}s")

    ## 3. Fragment the SMILES
    file_fragdb = Smiles_fragmentation(folderName_tmp, file_smi)
    time_3 = time.time()
    print(f"\t==> SMILES Fragmentation completed, costing time {int(time_3-time_2)}s")

    ## 4. Indexing to find the MMPs in the fragment file & Load the activity/property data
    file_mmpdb = Index_LinkActivity(folderName_tmp, file_fragdb, file_prop_csv)
    time_4 = time.time()
    print(f"\t==> MMPs DB analysis completed, costing time {int(time_4-time_3)}s")

    ## 5. Extracing all tables from database file
    dataDict_tables = DataExtractionFromDB(file_mmpdb, folderName_tmp)
    time_5 = time.time()
    print(f"\t==> Extracting data from MMPs DB completed, costing time {int(time_5-time_4)}s")

    ## 6. Clean up the MMPs data from the DB
    dataDict = MMPs_DataClean(dataDict_tables, add_symetric=True)

    ## ------------------ save the results  ------------------##
    dataTable = pd.DataFrame.from_dict(dataDict).T
    dataTable['Num_Consts'] = dataTable['pair_detail'].apply(lambda x: len(x))
    dataTable = dataTable.sort_values(by=["pair_id", "pair_info"], ascending=[True, True]).reset_index(drop=True)

    col_basic = ['KT_id_1', 'Smiles_1', 'KT_id_2', 'Smiles_2']
    col_bioassay = [f"{x}_1" for x in colName_prop_list] + [f"{x}_2" for x in colName_prop_list]
    col_pairinfo = ['pair_info', 'pair_id', 'Num_Consts', 'pair_detail']
    dataTable = dataTable[col_basic + col_bioassay + col_pairinfo]

    
    dataTable.to_csv(f"{fileName_out}.csv", index=False)
    print(f"7. The output file is saved to fileName_out")
    print("==> Entire analysis costs time = %ds ................" % (time.time()-beginTime))


if __name__ == '__main__':
    main()


