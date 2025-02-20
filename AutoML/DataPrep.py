'''
#Example for running the code
!/mnt/data0/Research/0_Test/cx_pKa/bash2py_yjing_local.bash 
python 
./DataPrep.py 
--i ./0_Data/DataView_MDCK_MDR1__Permeability_1__export.csv 
-d ',' 
--detectEncoding 
--colId 'Compound Name' 
--colSmi 'Structure' 
--colAssay 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)' 
--colAssayMod 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)' 
--colPreCalcDesc "ADME MDCK(WT) Permeability;Concat;Comments,Concat;Project,fake_column_name" 
-o './results/data_input_clean.csv'
'''

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

################################################################################################
##################################### CSV load Tools ###########################################
################################################################################################
## ---------------- detect encoding ----------------
def _determine_encoding(fileNameIn, default='utf-8'):
    try:
        # Step 1: Open the file in binary mode
        with open(fileNameIn, 'rb') as f:
            data = f.read()
            
        # Step 2: Detect the encoding using the chardet library
        import chardet
        encoding_result = chardet.detect(data)

        # Step 3: Retrieve the encoding information
        encoding = encoding_result['encoding']
    except Exception as e:
        print(f"\tError! Can not detect encoding, error {e}")
        encoding = default
    else:
        if encoding != default:
            print(f"\tUsing Encoding <{encoding}>")
    return encoding

## ---------------- load csv using pandas ----------------
def load_csv(fileNameIn, sep, detect_encoding=False):
    import pandas as pd

    print(f"\t==>Now reading the csv...")
    try:
        ## define encoding
        if detect_encoding:
            encoding = _determine_encoding(fileNameIn, default='utf-8')
        else:
            encoding = 'utf-8'
        ## load csv 
        dataTable = pd.read_csv(fileNameIn, sep=sep, encoding=encoding)
        print(f"\tThe original csv file has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns") 
    except Exception as e:
        print(f"\tError! Cannot load csv. errmsg: {e}")
        dataTable = None
    return dataTable

## ---------------- clean csv by rm NaNs and dups ----------------
def clean_csv(dataTable, cols_basic, cols_data, cols_mod=None):
    print(f"\t==>Now cleanning the csv...")
    ## remove NaN on essential cols
    for col in cols_basic+cols_data:
        assert col in dataTable.columns, f"\tError! Column <{col}> is not in the table!"
    dataTable = dataTable.dropna(subset=cols_basic+cols_data)
    print(f"\tAfter removing NaNs, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns")
    
    ## remove Duplicates on essential
    dataTable = dataTable.drop_duplicates(subset=cols_basic)
    print(f"\tAfter removing duplicates, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns")

    ## 
    if cols_mod is not None:
        for col_mod in cols_mod:
            assert col_mod in dataTable.columns, f"\tError! Operator column <{col}> is not in the table!"
            dataTable = dataTable[dataTable[col_mod].isin(['='])]
    print(f"\tAfter filterring operators, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns")

    dataTable = dataTable.reset_index(drop=True)
    return dataTable

## ---------------- extract custom descriptors ----------------
def extract_custom_desc(dataTable, colName_mid, colName_custom_desc):
    if colName_custom_desc is not None:
        print(f"\tNow extracting the custom desc using the defined column names: {colName_custom_desc}")
        list_custom_desc = colName_custom_desc.split(',')
        list_available_desc = []
        for desc in list_custom_desc:
            if desc in dataTable.columns:
                list_available_desc.append(desc)
            else:
                print(f"\t\tWarning! This custom descriptor <{desc}> is not in the data table, so ignored this column")
        print(f"\tThere are total {len(list_available_desc)} custom descriptors extracted")

        if len(list_available_desc) > 0:
            dataTable_desc = dataTable[[colName_mid]+list_available_desc]
            dataTable_desc = dataTable_desc.rename(columns={col: f"custDesc_{col}" for col in list_available_desc})
            dataTable = dataTable.drop(columns=list_available_desc)
            print(f'\tThe custom desciptor table has <{dataTable_desc.shape[0]}> rows and <{dataTable_desc.shape[1]}> columns')
            print(f'\tAfter extracting the custom desc, the table has <{dataTable_desc.shape[0]}> rows and <{dataTable_desc.shape[1]}> columns')
        else:
            dataTable_desc = None
    else:
        print(f"\tNo custom desc is defined")
    return dataTable, dataTable_desc

################################################################################################
################################## SMILES processing Tools #####################################
################################################################################################
## ---------------- clean up smiles ----------------
def _mute_warning_rdkit():
    from rdkit import RDLogger
    # Get the RDKit logger
    lg = RDLogger.logger()
    # Set the logging level to CRITICAL to mute warnings and other lower-level messages
    lg.setLevel(RDLogger.CRITICAL)
    return None

def _cleanUpSmiles(smi, canonical=True, errmsg=False):
    from rdkit import Chem
    try:
        ## remove extended Smiles
        if "|" in smi:
            smi = smi.split("|")[0]
        ## remove \
        if "\\" in smi:
            smi = smi.replace('\\', '\\\\')
            # print(f'\tThere is a "\\" in the SMILES. Adding 1 "\\" into the SMILES, now new SMILES is {smi}')
        ## strp salts
        if '.' in smi:
            smi = max(smi.split('.'), key=len)
        ## remove next line
        smi = smi.replace("\n", "").replace("\r", "").replace("\r\n", "").replace(" ", "")
        ## rdkit smiles vadality checking
        _mute_warning_rdkit()
        mol = Chem.MolFromSmiles(smi)
        smi_rdkit = Chem.MolToSmiles(mol, canonical=canonical)
    except Exception as e:
        print(f"\t\tWarning! Cannot prase this Smiles: <{smi}>")
        if errmsg:
            print(f"\t\tErrMsg: {e}")
        import numpy as np
        smi_rdkit = np.nan
    return smi_rdkit

def clean_smiles(dataTable, colName_smi, canonical=True, errmsg=False):
    print(f"\t==>Now cleanning the Smiles...")
    assert colName_smi in dataTable.columns, f"\tError! Column <{colName_smi}> is not in the table!"
    dataTable[f"{colName_smi}_original"] = dataTable[colName_smi]
    dataTable[colName_smi] = dataTable[colName_smi].apply(lambda x: _cleanUpSmiles(x, canonical=canonical, errmsg=errmsg))
    return dataTable

################################################################################################
################################## main #####################################
################################################################################################
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--detectEncoding', action="store_true", help='detect the encoding type of the csv file')

    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Structure', help='The column name of the compound smiles')

    parser.add_argument('--colAssay', action="store", default='IC50', help='The column names of the assay values, only 1 column is accepted')
    parser.add_argument('--colAssayMod', action="store", default=None, help='The column names of the assay values operator, only 1 column is accepted')

    parser.add_argument('--colPreCalcDesc', action="store", default=None, help='comma separated string e.g., <desc_1,desc_2,desc_3>')

    parser.add_argument('-o', '--output', action="store", default="./results/data_input_clean.csv", help='save the cleaned csv file')

    args = parser.parse_args()
    return args

def main():
    args = Args_Prepation(parser_desc='Preparing the input files')

    fileNameIn = args.input    # f"./0_Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
    sep = args.delimiter    # ','
    detect_encoding = True if args.detectEncoding else False

    colName_mid = args.colId    # 'Compound Name'
    colName_smi = args.colSmi    # 'Structure'

    colName_expt = args.colAssay    #'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
    colName_expt_operator = args.colAssayMod    # 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'    ## None

    colName_custom_desc = args.colPreCalcDesc

    filePathOut = args.output    ## 'Concat;Project'   

    dataTable = load_csv(fileNameIn, sep=sep, detect_encoding=detect_encoding)
    dataTable = clean_smiles(dataTable, colName_smi=colName_smi, canonical=False, errmsg=False)
    dataTable = clean_csv(dataTable, cols_basic=[colName_mid, colName_smi], cols_data=[colName_expt], cols_mod=[colName_expt_operator])

    dataTable_y = dataTable[[colName_mid, colName_expt]]
    dataTable, dataTable_desc = extract_custom_desc(dataTable, colName_mid, colName_custom_desc)

    ## save output
    import os
    output_folder = os.path.dirname(filePathOut)
    os.makedirs(output_folder, exist_ok=True)
    dataTable.to_csv(filePathOut, index=False)
    print(f"\tThe cleaned data table has been saved to {filePathOut}")
    
    ## save the y output
    ofileName_y = os.path.join(output_folder, f'outcome_expt.csv')
    dataTable_y.to_csv(ofileName_y, index=False)
    print(f"\tThe experiment outcome table has been saved to {ofileName_y}")

    ## save the desc output
    if dataTable_desc is not None:
        ofileName_desc = os.path.join(output_folder, f'descriptors_custom.csv')
        dataTable_desc.to_csv(ofileName_desc, index=False)
        print(f"\tThe custom descriptors table has been saved to {ofileName_desc}")

if __name__ == '__main__':
    main()