'''
--input     './results/data_input_clean.csv'
--delimiter     ','
--colId     'Compound Name'
--colSmi    'Structure'
--colDate   "ADME MDCK(WT) Permeability;Concat;Run Date"
--split'    'random'
--CV   10
--rng   666666
--hasVal    True
--output    "outputfolder"
'''


## Suppress RDKit warnings
def mute_rdkit():
    from rdkit import RDLogger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


## ================================================================================================
## ==================================== random split ==============================================
## ================================================================================================
def nFoldSplit_random(dataTable, colName_mid='Compound Name', CV=10, rng=666666, hasVal=True):
    ds_size = dataTable.shape[0]
    assert CV*2 < ds_size, f"\tError, the dataset (N={ds_size}) is too small to do a {CV}_fold split! Please decrease the CV value ({CV})\n"

    dataTable_split = dataTable[[colName_mid]].reset_index(drop=True)
    list_mol_idx = dataTable_split.index.to_numpy()

    # Shuffle the list using random seed
    import numpy as np
    np.random.seed(rng)
    np.random.shuffle(list_mol_idx)

    # Split the list into N sublists
    sublists = np.array_split(list_mol_idx, CV)
    idx_test = sublists[CV-1]
    idx_val = sublists[CV-2] if hasVal else []
    idx_train = [i for i in list_mol_idx if i not in idx_test and i not in idx_val]
    print(f"\tSplit the data (n={len(list_mol_idx)}) into Train({len(idx_train)}), Val({len(idx_val)}), and Test({len(idx_test)})\n")

    # Apply the function to assign values to the new column 'A'
    dataTable_split[f'Split'] = dataTable_split.index.to_series().apply(lambda x: assign_value(x, idx_train, idx_val, idx_test))
    return dataTable_split


## ================================================================================================
## ==================================== temporal split ============================================
## ================================================================================================
def nFoldSplit_temporal(dataTable, colName_mid='Compound Name', colName_date="Created On", CV=10, hasVal=True):
    ds_size = dataTable.shape[0]
    assert CV*2 < ds_size, f"\tError, the dataset (N={ds_size}) is too small to do a {CV}_fold split! Please decrease the CV value ({CV})\n"

    dataTable_split = dataTable[[colName_mid, colName_date]].reset_index(drop=True)
    try:
        import pandas as pd
        dataTable_split[colName_date] = dataTable_split[colName_date].str.split(';').str[0]
        dataTable_split["date_formatted"] = pd.to_datetime(dataTable_split[colName_date])
        dataTable_split = dataTable_split.sort_values(by=["date_formatted"], ascending=[True])
    except Exception as e:
        print(f"\tWarning! The mol date column <{colName_date}> cannot be formatted. Error mgs: {e}\n")
    else:
        # Split the list into N sublists
        import numpy as np
        list_mol_idx = dataTable_split.index.to_numpy()
        try:
            sublists = np.array_split(list_mol_idx, CV)
        except Exception as e:
            print(f"\tWarning! Cannot split data based on date. Error mgs: {e}\n")
        else:
            idx_test = sublists[CV-1]
            idx_val = sublists[CV-2] if hasVal else []
            idx_train = [i for i in list_mol_idx if i not in idx_test and i not in idx_val]
            print(f"\tSplit the data (n={len(list_mol_idx)}) into Train({len(idx_train)}), Val({len(idx_val)}), and Test({len(idx_test)})\n")
            
            # Apply the function to assign values to the new column 'A'
            dataTable_split[f'Split'] = dataTable_split.index.to_series().apply(lambda x: assign_value(x, idx_train, idx_val, idx_test))
    return dataTable_split 


## ================================================================================================
## ==================================== diverse split ============================================
## ================================================================================================
def nFoldSplit_diverse(dataTable, colName_mid='Compound Name', colName_smi="Structure", CV=10, rng=666666, hasVal=True):
    ds_size = dataTable.shape[0]
    assert CV*2 < ds_size, f"\tError, the dataset (N={ds_size}) is too small to do a {CV}_fold split! Please decrease the CV value ({CV})\n"

    dataTable_split = dataTable[[colName_mid, colName_smi]].sample(frac=1).reset_index(drop=True)
    smiles_list = dataTable_split[colName_smi].to_list()

    ## calc the fps
    mute_rdkit()
    import numpy as np
    from rdkit import Chem, DataStructs, SimDivFilters
    from rdkit.Chem import AllChem
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 3, nBits=2048) for smi in smiles_list]

    ## Generate the distance matrix in advance
    ds=[]
    for i in range(1,len(fps)):
        ds.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))

    ## Initialize the MaxMinPicker
    picker = SimDivFilters.MaxMinPicker()

    ## define the number of mols to pick for test/validation
    num_picks = int(ds_size/CV)
    num_picks_real = 2*num_picks if hasVal else num_picks

    ## Select N diverse molecules from the set
    pick_idx = picker.Pick(np.array(ds), len(fps), num_picks_real, seed=rng)
    idx_test = pick_idx[:num_picks] if hasVal else pick_idx
    idx_val = pick_idx[num_picks:] if hasVal else []
    idx_train = [i for i in dataTable_split.index if i not in pick_idx]
    print(f"\tSplit the data (n={len(fps)}) into Train({len(idx_train)}), Val({len(idx_val)}), and Test({len(idx_test)})\n")

    # Apply the function to assign values to the new column 'A'
    dataTable_split[f'Split'] = dataTable_split.index.to_series().apply(lambda x: assign_value(x, idx_train, idx_val, idx_test))

    return dataTable_split



################################################################################################
################################## main #####################################
################################################################################################
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument('-i', '--input', action="store", default="./Results/data_input_clean.csv", help='The input csv file')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Structure', help='The column name of the compound smiles')
    parser.add_argument('--colDate', action="store", default=None, help='The column names of the date')
    parser.add_argument('-s', '--split', action="store", default='random', help='The split method. select from [random, temporal, diverse]')
    parser.add_argument('--cols', action="store", default='Split', help='The column name of the split')


    parser.add_argument('--CV', action="store", default="10", help='The Cross-validation fold for split')
    parser.add_argument('--rng', action="store", default="666666", help='The random seed of random selection')
    parser.add_argument('--hasVal', action="store", default="True", help='If separate a validation set')

    parser.add_argument('-o', '--output', action="store", default="./Results/data_split.csv", help='save the splited csv file')

    args = parser.parse_args()
    return args

## assign values (dataset) to the datat table
def assign_value(idx, list_train, list_val, list_test):
    if idx in list_train:
        return 'Training'
    elif idx in list_val:
        return 'Validation'
    elif idx in list_test:
        return 'Test' 
    else:
        return 'Training'


## 
def run_script(fileNameIn, sep=',', colName_mid='Compound Name', colName_smi='Structure',  split_method='random', 
               colName_date='Created On',CV=10, rng=666666, hasVal=True, filePathOut="./Results/data_split.csv"):
    print(f">>>>Spliting dataset ...\n")

    ## ------------ load data ------------
    import pandas as pd
    dataTable_raw = pd.read_csv(fileNameIn, sep=sep)
    print(f"\t{dataTable_raw.shape}")
    assert colName_mid in dataTable_raw.columns, f"\tColumn name for mol ID <{colName_mid}> is not in the table.\n"

    print(f"\tData split method: {split_method}")
    if split_method not in ['random', 'temporal', 'diverse']:
        print(f"\tWarning, the split method should be selected from [random, temporal, diverse]\n")
        split_method = 'random'
        print(f"\tUse <random> instead\n")
    ## ------------ calculate rdkit properties ------------
    if split_method == 'random':
        dataTable_split = nFoldSplit_random(dataTable_raw, colName_mid, CV=CV, rng=rng, hasVal=hasVal)

    ## ------------ calculate mol fingerprints ------------
    elif split_method == 'temporal':
        assert colName_date is not None, f"\tColumn name for date <{colName_date}> should not be None when using {split_method} split\n"
        assert colName_date in dataTable_raw.columns, f"\tColumn name for date <{colName_date}> should be in the table column when using {split_method} split\n"
        dataTable_split = nFoldSplit_temporal(dataTable_raw, colName_mid, colName_date, CV=CV, hasVal=hasVal)

    ## ------------ calculate chemAxon properties ------------
    elif split_method == 'diverse':
        assert colName_smi in dataTable_raw.columns, f"\tColumn name for mol smiles <{colName_smi}> is not in the table.\n"
        dataTable_split = nFoldSplit_diverse(dataTable_raw, colName_mid, colName_smi, CV=CV, rng=rng, hasVal=hasVal)

    ## ------------ save the split ------------
    import os
    folderPathOut = os.path.dirname(filePathOut)    ## './results'
    os.makedirs(folderPathOut, exist_ok=True)
        
    dataTable_split.to_csv(filePathOut, index=False)
    print(f"\tThe cleaned data table has been saved to {filePathOut}\n")
    return filePathOut


def main():
    print(f">>>>Spliting dataset ...\n")
    args = Args_Prepation(parser_desc='Preparing the input files and the descriptors')
    fileNameIn = args.input    # '../../1_DataPrep/results/data_input_clean.csv'
    sep = args.delimiter 
    colName_mid = args.colId    # 'Compound Name'
    colName_smi = args.colSmi    # 'Structure'
    colName_date = args.colDate    # 'Created On'

    split_method = args.split
    CV = int(args.CV)
    rng = int(args.rng)
    hasVal = True if args.hasVal in ['TRUE', 'True', 'true', 'YES', 'Yes', 'yes'] else False

    filePathOut = args.output


    ## run code
    filePathOut_split = run_script(fileNameIn, sep, colName_mid, colName_smi, split_method, colName_date, CV, rng, hasVal, filePathOut)

if __name__ == '__main__':
    main()