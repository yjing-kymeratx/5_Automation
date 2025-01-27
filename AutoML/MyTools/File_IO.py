## 
import os
import pandas as pd

## --------------------- determine the encoding ---------------------
def _determine_encoding(fileNameIn, default='utf-8'):
    import chardet

    try:
        # Step 1: Open the file in binary mode
        with open(fileNameIn, 'rb') as f:
            data = f.read()
            
        # Step 2: Detect the encoding using the chardet library
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


## --------------------- read csv ---------------------
def read_csv_file(fileName_in, sep=','):
    assert os.path.exists(fileName_in), f"File {fileName_in} does not exist"
    try:
        ## determine encoding type
        encoding = _determine_encoding(fileName_in)
        ## read csv file
        print(f"\tNow pandas reading csv data using <{encoding}> encoding from {fileName_in}")
        dataTable = pd.read_csv(fileName_in, sep=sep, encoding=encoding).reset_index(drop=True)
    except Exception as e:
        dataTable = None
        print(f'\tError: cannot read output file {fileName_in}; error msg: {e}')
    else:
        print(f"\tThe loaded raw data has <{dataTable.shape[0]}> rows and {dataTable.shape[1]} columns")
    return dataTable


## --------------------- create folder ---------------------
def FolderCreator(my_folder=None):
    ## ------- simply clean up the folder path -------
    if my_folder is None:
        my_folder='./tmp'
    
    elif '/' not in my_folder:
        my_folder = os.path.join(os.getcwd(), my_folder)

    ## ------- Check if the folder exists -------
    if not os.path.isdir(my_folder):
        os.makedirs(my_folder)
        print(f"\tCreated folder: {my_folder}")
    else:
        print(f'\t{my_folder} is existing')
    return my_folder



