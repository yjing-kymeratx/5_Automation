import numpy as np
import pandas as pd

##############################################################################################
##################################### Custom Tools ###########################################
##############################################################################################

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
def read_csv(fileNameIn, sep, detect_encoding=False):
    print(f"\t==>Now reading the csv...")
    try:
        ## define encoding
        if detect_encoding:
            encoding = _determine_encoding(fileNameIn, default='utf-8')
        else:
            encoding = 'utf-8'
        ## load csv
        
        dataTable = pd.read_csv(fileNameIn, sep=sep, encoding=encoding)
        print(f"\tThe original csv file has {dataTable.shape[0]} rows and {dataTable.shape[1]} columns") 
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
    print(f"\tAfter removing NaNs, the table has {dataTable.shape[0]} rows and {dataTable.shape[1]} columns")
    
    ## remove Duplicates on essential
    dataTable = dataTable.drop_duplicates(subset=cols_basic)
    print(f"\tAfter removing duplicates, the table has {dataTable.shape[0]} rows and {dataTable.shape[1]} columns")

    ## 
    if cols_mod is not None:
        for col_mod in cols_mod:
            assert col_mod in dataTable.columns, f"\tError! Operator column <{col}> is not in the table!"
            dataTable = dataTable[dataTable[col_mod].isin(['='])]
    print(f"\tAfter filterring operators, the table has {dataTable.shape[0]} rows and {dataTable.shape[1]} columns")

    dataTable = dataTable.reset_index(drop=True)
    return dataTable

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
        smi_rdkit = np.nan
    return smi_rdkit

def clean_smiles(dataTable, colName_smi, canonical=True, errmsg=False):
    print(f"\t==>Now cleanning the Smiles...")
    assert colName_smi in dataTable.columns, f"\tError! Column <{colName_smi}> is not in the table!"
    dataTable[f"{colName_smi}_original"] = dataTable[colName_smi]
    dataTable[colName_smi] = dataTable[colName_smi].apply(lambda x: _cleanUpSmiles(x, canonical=canonical, errmsg=errmsg))
    return dataTable

##############################################################################################
############################################ main ############################################
##############################################################################################