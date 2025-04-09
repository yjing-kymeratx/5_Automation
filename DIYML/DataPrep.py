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
##################################### generalTools ###########################################
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
        print(f"\tError! Can not detect encoding, error {e}\n")
        encoding = default
    else:
        if encoding != default:
            print(f"\tUsing Encoding <{encoding}>.\n")
    return encoding

## ---------------- split operators ----------------
def findOperator(dataValue):
    import numpy as np
    data_mod, data_num = "=", np.nan
    try:
        dataString = str(dataValue).replace(' ', '')
        for opt in ['>=', '<=', '>', '<']:
            if opt in dataString:
                data_mod = opt
                data_num = dataString.replace(opt, '')
                break
            else:
                data_num = dataString
        data_num = float(data_num)
    except Exception as e:
        print(f"\t\tCannot numerical this value {dataValue}. Error msg: {e}")
    return data_mod, data_num


################################################################################################
##################################### CSV load Tools ###########################################
################################################################################################


## ---------------- load csv using pandas ----------------
def load_csv(fileNameIn, sep, detect_encoding=False):
    import pandas as pd

    print(f"\t==>Now reading the csv...\n")
    try:
        ## define encoding
        if detect_encoding:
            encoding = _determine_encoding(fileNameIn, default='utf-8')
        else:
            encoding = 'utf-8'
        ## load csv 
        dataTable = pd.read_csv(fileNameIn, sep=sep, encoding=encoding)
        print(f"\tThe original csv file has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns\n") 
    except Exception as e:
        print(f"\tError! Cannot load csv. errmsg: {e}\n")
        dataTable = None
    return dataTable

## ---------------- clean csv by rm NaNs and dups ----------------
def clean_csv(dataTable, cols_basic, col_y=None, col_ymod=None, cleanedData=True):
    print(f"\t==>Now cleanning the csv...")
    ## ------------ remove NaN on essential cols
    for col in cols_basic:
        assert col in dataTable.columns, f"\tError! Column <{col}> is not in the table!\n"
    dataTable = dataTable.dropna(subset=cols_basic)
    print(f"\tAfter removing NaNs in {cols_basic}, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns\n")
    
    ## -------- remove Duplicates on essential
    dataTable = dataTable.drop_duplicates(subset=cols_basic)
    print(f"\tAfter removing duplicates, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns\n")

    ## -------- clean the y columns if there is operator in the column
    print(f"\tCleaning up the y columns: col_y={col_y}; col_ymod={col_ymod}; cleanedData={cleanedData}")
    dataTable[f"{col_y}_original"] = dataTable[col_y].apply(lambda x: x)

    if not cleanedData:
        col_ymod = f"{col_y}_MOD" if col_ymod is None else col_ymod
        dataTable[col_ymod] = dataTable[col_y].apply(lambda x: findOperator(x)[0])
        dataTable[col_y] = dataTable[col_y].apply(lambda x: findOperator(x)[1])
    try:
        dataTable[col_y] = dataTable[col_y].astype(float)
    except Exception as e:
        print(f"\t\tWarning! The experimental outcome column <{col_y}> cannot be transfer to numerical data. Error msg: {e}")

    ##
    if col_y is not None:
        assert col_y in dataTable.columns, f"\tError! expterimental(y) column <{col_y}> is not in the table!\n"
        dataTable = dataTable.dropna(subset=col_y)
        print(f"\tAfter removing NaNs in {col_y}, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns\n")

        if col_ymod is not None:
            assert col_ymod in dataTable.columns, f"\tError! Operator column <{col_ymod}> is not in the table!\n"
            dataTable = dataTable[dataTable[col_ymod].isin(['='])]
    
    print(f"\tAfter filterring operators, the table has <{dataTable.shape[0]}> rows and <{dataTable.shape[1]}> columns\n")
    dataTable = dataTable.reset_index(drop=True)
    return dataTable

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

def cleanUpSmiles(smi, canonical=True, errmsg=False):
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
        print(f"\t\tWarning! Cannot prase this Smiles: <{smi}>\n")
        if errmsg:
            print(f"\t\tErrMsg: {e}\n")
        import numpy as np
        smi_rdkit = np.nan
    return smi_rdkit

def clean_smiles(dataTable, colName_smi, canonical=True, errmsg=False):
    print(f"\t==>Now cleanning the Smiles...\n")
    # print(list(dataTable.columns))
    assert colName_smi in dataTable.columns, f"\tError! Column <{colName_smi}> is not in the table!\n"
    dataTable[f"{colName_smi}_original"] = dataTable[colName_smi]
    dataTable[colName_smi] = dataTable[colName_smi].apply(lambda x: cleanUpSmiles(x, canonical=canonical, errmsg=errmsg))
    return dataTable

################################################################################################
################################## main #####################################
################################################################################################
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns (currently only use comma and tab)')
    parser.add_argument('--detectEncoding', action="store", default='True', help='detect the encoding type of the csv file')

    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Structure', help='The column name of the compound smiles')

    parser.add_argument('--colAssay', action="store", default=None, help='The column names of the assay values, max 1 column is accepted')
    parser.add_argument('--colAssayMod', action="store", default=None, help='The column names of the assay values operator, only 1 column is accepted')
    parser.add_argument('--cleanData', action="store", default='True', help='The column names of the assay values operator, only 1 column is accepted')

    parser.add_argument('-o', '--output', action="store", default="./Results/data_input_clean.csv", help='save the cleaned csv file')
    parser.add_argument('-oy', '--outputy', action="store", default="./Results/outcome_expt.csv", help='save the expert outcome csv file')

    args = parser.parse_args()
    return args

##
def run_script(fileNameIn, sep=',', detect_encoding=True, colName_mid='Compound Name', colName_smi='Structure', colName_expt='IC50_uM', 
               colName_expt_operator=None, cleanedData=True, filePathOut="./Results/data_input_clean.csv", ofileName_y="./Results/outcome_expt.csv"):
    print(f">>>>Preparing dataset ...\n")
    import time
    beginTime = time.time()

    import os
    assert os.path.exists(fileNameIn), f"Error! The input file {fileNameIn} is not existing.\n"
    ## ---------- read data ----------
    dataTable = load_csv(fileNameIn, sep=sep, detect_encoding=detect_encoding)

    ## ---------- clean smiles ----------
    dataTable = clean_smiles(dataTable, colName_smi=colName_smi, canonical=False, errmsg=False)
    dataTable = clean_csv(dataTable, cols_basic=[colName_mid, colName_smi], col_y=colName_expt, col_ymod=colName_expt_operator, cleanedData=cleanedData)
        
    ## save output
    import os
    folderPathOut = os.path.dirname(filePathOut)
    os.makedirs(folderPathOut, exist_ok=True)
    dataTable.to_csv(filePathOut, index=False)
    print(f"\tThe cleaned data table has been saved to {filePathOut}\n")

    ## save the y output
    if colName_expt is not None:
        dataTable_y = dataTable[[colName_mid, colName_expt]]        
        # ofileName_y = os.path.join(folderPathOut, f'outcome_expt.csv')

        dataTable_y.to_csv(ofileName_y, index=False)
        print(f"\tThe experiment outcome table has been saved to {ofileName_y}\n")

    print(f">>>>The Data Preparation process takes {round(time.time()-beginTime, 2)} sec")
    return filePathOut, ofileName_y


##
def main():
    args = Args_Prepation(parser_desc='Preparing the input files')

    fileNameIn = args.input    # f"./0_Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
    sep = args.delimiter    # ','
    detect_encoding = True if args.detectEncoding in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False

    colName_mid = args.colId    # 'Compound Name'
    colName_smi = args.colSmi    # 'Structure'

    colName_expt = args.colAssay    #'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
    colName_expt_operator = args.colAssayMod    # 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'    ## None

    filePathOut = args.output    ## 'Concat;Project' 
    ofileName_y = args.outputy

    cleanedData = True if args.cleanData in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False
    # cleanedData = False

    ## run the script
    filePathOut, ofileName_y = run_script(fileNameIn, sep, detect_encoding, colName_mid, colName_smi, colName_expt, colName_expt_operator, cleanedData, filePathOut, ofileName_y)

if __name__ == '__main__':
    main()