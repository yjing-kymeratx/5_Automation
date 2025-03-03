'''
-i './Data/DataView_MDCK_MDR1__Permeability_1__export_top30.csv'
--inputType 'csv'
--delimiter ','
--colId 'Compound Name'
--colSmi 'Structure'
--colAssay 'ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
--modelFile f"{folderPathOut_model}/{ml_methed}_models.pickle"
'''

# Mute warining
import warnings
warnings.filterwarnings("ignore")

import sys
dataDir_AutoML = "/mnt/data0/Research/5_Automation/AutoML"
sys.path.append(dataDir_AutoML)

## =================================================================================
## ================= read csv file and extract molecular information ===============
## =================================================================================
def load_csv_file(fileNameIn, colName_mid, colName_smi, custDesc_list=[], colName_expt=None, sep=','):
    import DataPrep
    import numpy as np
    import pandas as pd
    
    ## ----------- load table from csv file -----------
    dataTable_raw = pd.read_csv(fileNameIn, sep=sep).reset_index(drop=True)
    print(f"\t\tThe input file has {dataTable_raw.shape[0]} rows and {dataTable_raw.shape[1]} columns")

    ## ----------- check columns -----------
    col_list_essential = [colName_mid, colName_smi]
    if colName_expt is not None:
        col_list_essential.append(colName_expt)

    ## check custom descriptors
    if len(custDesc_list) > 0:
        for custDesc in custDesc_list:
            custDesc_raw = custDesc.replace('custDesc_', '')
            assert custDesc_raw in dataTable_raw.columns, print(f"\t\tError, custom descriptor column <{custDesc_raw}> is missing")
            dataTable_raw = dataTable_raw.rename(columns={custDesc_raw: custDesc})

    for col in col_list_essential:
        assert col in dataTable_raw.columns, print(f"\t\tError, column <{col}> is missing")

    ## ----------- extract data -----------
    mol_dict = {}
    for idx in dataTable_raw.index:
        mol_dict[idx] = {}
      
        ## ----------- molecular name/id -----------
        mid = dataTable_raw[colName_mid][idx] if dataTable_raw[colName_mid].notna()[idx] else f"Unamed_mol_row_{idx}"
        mol_dict[idx][colName_mid] = mid

        ## ----------- smiles -----------
        if dataTable_raw[colName_smi].notna()[idx]:
            smi = dataTable_raw[colName_smi][idx]
            smi_clean = DataPrep.cleanUpSmiles(smi)
            mol_dict[idx][colName_smi] = smi_clean
        else:
            mol_dict[idx][colName_smi] = np.nan

        ## ----------- expt(y) ----------- 
        if colName_expt is not None:
            if dataTable_raw[colName_expt].notna()[idx]:
                mol_dict[idx][colName_expt] = dataTable_raw[colName_expt][idx]

        # ----------- custom desc --------------
        if len(custDesc_list) > 0:
            for custDesc in custDesc_list:
                if dataTable_raw[custDesc].notna()[idx]:
                    mol_dict[idx][custDesc] = float(dataTable_raw[custDesc][idx])
                else:
                    mol_dict[idx][custDesc] = np.nan
    return mol_dict

##
def load_sdf_file(fileNameIn, colName_mid, colName_smi, custDesc_list=[], colName_expt=None):
    ## ----------- load table from sdf file -----------
    # import numpy as np
    from rdkit import Chem

    ## ----------- load table from sdf file -----------
    supplier = Chem.SDMolSupplier(fileNameIn)
    print(f"\tThe input sdf file has {len(supplier)} mols")

    ## ----------- extract data -----------
    mol_dict = {}
    for idx in range(len(supplier)):
        mol = supplier[idx]
        if mol is not None:            
            mol_dict[idx] = {}
            ## ----------- molecular name/id -----------
            mid = mol.GetProp(colName_mid) if mol.HasProp(colName_mid) else f"Unamed_mol_row_{idx}"
            mol_dict[idx][colName_mid] = mid

            ## ----------- smiles -----------
            try:
                smiles = Chem.MolToSmiles(mol)
            except:
                mol_dict[idx][colName_smi] = None
                print(f"\t\tError, cannot generate smiles for mol <{mid}>")
            else:
                mol_dict[idx][colName_smi] = smiles    #, canonical=True

            ## ----------- expt(y) -----------
            if colName_expt is not None:
                if mol.HasProp(colName_expt):
                    mol_dict[idx][colName_expt] = mol.GetProp(colName_expt)
                else:
                    print(f"\t\tError, expt value column <{custDesc_raw}> is missing for mol <{mid}>")

            # ----------- custom desc --------------
            if len(custDesc_list) > 0:
                for custDesc in custDesc_list:
                    custDesc_raw = custDesc.replace('custDesc_', '')
                    if mol.HasProp(custDesc_raw):
                        mol_dict[idx][custDesc] = float(mol.GetProp(custDesc_raw))
                    else:
                        # mol_dict[idx][custDesc] = np.nan
                        print(f"\t\tError, custom descriptor column <{custDesc}> is missing for mol <{mid}>")
    print(f"\tThere are total <{len(mol_dict)}> mols loaded into mol_dict.")
    return mol_dict


## ======================================================================================================
## ================= determine the descriptor types/calculators needs to be calculated =================
## ======================================================================================================
def get_desc_descriptors(desc_list, desc_calc_param):
    print(f"\t\tDescriptors (N={len(desc_list)}) used for model building&predicting are: {desc_list}")
    desc_type_list, desc_calculator_list, custDesc_list = [], [], []

    ## find desc types
    for desc in desc_list:
        desc_prefix = desc.split('_')[0]
        if desc_prefix not in desc_type_list:
            desc_type_list.append(desc_prefix)
        if desc_prefix == 'custDesc':
            custDesc_list.append(desc)

    ## rdkit prop
    if 'rd' in desc_type_list:
        from AutoML.DescGen_backup import desc_calculator_rdkit
        physChem, subStr, clean = desc_calc_param['rd_physChem'], desc_calc_param['rd_subStr'], desc_calc_param['rd_clean']
        calculator_rd = desc_calculator_rdkit(physChem=physChem, subStr=subStr, clean=clean)
        desc_calculator_list.append(calculator_rd)
        
    ## mol fp
    if 'fp' in desc_type_list:
        from AutoML.DescGen_backup import desc_calculator_morganFPs
        calculator_fp = desc_calculator_morganFPs(radius=desc_calc_param['fp_radius'], nBits=desc_calc_param['fp_nBits'])
        desc_calculator_list.append(calculator_fp)

    ## chemxxon prop
    if 'cx' in desc_type_list:
        from AutoML.DescGen_backup import desc_calculator_chemaxon
        calculator_cx = desc_calculator_chemaxon(version=desc_calc_param['cx_version'], desc_list=desc_calc_param['cx_desc'])
        desc_calculator_list.append(calculator_cx)

    if 'custDesc' in desc_type_list:
        pass
    
    print(f"\t\tThere is/are total {len(desc_calculator_list)} calculators prepared")
    return desc_calculator_list, custDesc_list

## ==============================================================================
## =========================== preparing the descriptors ========================
## ==============================================================================
def normalization_desc(desc_value, desc_norm_param, desc):
    ## doesn't normalize fingerprint binary data
    if desc[:3] == 'fp_' or desc not in desc_norm_param:
        norm_value = desc_value
    else:
        ## error check
        assert desc in desc_norm_param, f"\t\tError! descriptor <{desc}> is not in training normalization param file"
        norm_mean, norm_std = desc_norm_param[desc]["mean"], desc_norm_param[desc]["std"]
        try:
            norm_value = 0 if norm_std == 0 else (desc_value - norm_mean)/norm_std
        except Exception as e:
            print(f"\t\t=>Error! Cannot normalize <{desc}> with data <<{desc_value}>> , error msg:{e}")
            norm_value = desc_value
    return norm_value

def prep_desc(mol_dict, colName_smi, desc_list, desc_calculator_list, custDesc_list, desc_norm_param, desc_impu_param):
    import numpy as np
    import pandas as pd
    
    do_norm = True if len(desc_norm_param) > 0 else False
    do_impu = True if len(desc_impu_param) > 0 else False

    ## ----------- calculater descriptors -----------
    for idx in mol_dict:
        if colName_smi not in mol_dict[idx]:
            print(f"==>><{colName_smi}>@ <{mol_dict[idx]}>")
        smi = mol_dict[idx][colName_smi]
        if smi is not None:
            ## calculated desc
            for desc_calculator in desc_calculator_list:
                desc_calculator.calculate(smi)
                desc_calc_dict = desc_calculator.dataDict_results
                for desc in desc_calc_dict:
                    if desc in desc_list:
                        mol_dict[idx][desc] = desc_calc_dict[desc]

            for custDesc in custDesc_list:
                if custDesc not in mol_dict[idx]:
                    print(f"\t\tWarning! Custom descriptor <{custDesc}> is not in mol_dict for row {idx}")
                    mol_dict[idx][custDesc] = np.nan

    ## ----------- processing descriptors -----------
    for desc in desc_list:
        for idx in mol_dict:
            if desc in mol_dict[idx]:
                ## do normalization
                if do_norm:    
                        mol_dict[idx][desc] = normalization_desc(mol_dict[idx][desc], desc_norm_param, desc)
            else:
                ## imputation
                if not do_impu:
                    print(f"should")
    
                assert desc in desc_impu_param, f"\t\tError! descriptor <{desc}> is not in imputation param file"
                with pd.option_context('future.no_silent_downcasting', True):
                    mol_dict[idx][desc] = desc_impu_param[desc]["median"]

    ## transfer to dataframe
    dataTable_desc = pd.DataFrame.from_dict(mol_dict).T.dropna(subset=[colName_smi])
    dataTable_desc_raw = pd.DataFrame.from_dict(mol_dict).T.drop(columns=[colName_smi])
    print(f"\tThe processed descriptor table has shape of {dataTable_desc.shape}")

    return dataTable_desc, dataTable_desc_raw

## ==================

####################################################################
########################## Tools ###################################
####################################################################
## get the args
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)

    parser.add_argument('--inputType', action="store", default="csv", help='The input files type, can be [csv] or [sdf]')
    parser.add_argument('-i', '--input', action="store", default=None, help='The input files')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Structure', help='The column name of the compound smiles')
    parser.add_argument('--colAssay', action="store", default=None, help='The column names of the assay values, max 1 column is accepted')
    # parser.add_argument('--colPreCalcDesc', action="store", default=None, help='comma separated string e.g., <desc_1,desc_2,desc_3>')   
    
    parser.add_argument('--modelFile', action="store", default=None, help='The model pickle file')
    parser.add_argument('-o', '--output', action="store", default="./results.csv", help='the output csv file')

    args = parser.parse_args()
    return args

####################################################################
######################### main function ############################
####################################################################
def main():
    print(f">>>>Making predictions ...")
    ## ------------ load args ------------
    args = Args_Prepation(parser_desc='Feature selection')
    if True:
        inputType = args.inputType
        fileNameIn = args.input
        sep = args.delimiter    # ','
        
        colName_mid = args.colId 
        colName_smi = args.colSmi 
        colName_expt = args.colAssay
        # colPreCalcDesc = args.colPreCalcDesc 
        
        import os
        filePathOut = args.output
        folderPathOut = os.path.dirname(filePathOut)
        os.makedirs(folderPathOut, exist_ok=True)

        
        model_file = args.modelFile  
        ## ------------ load model from file ------------
        assert model_file is not None, f"\tError! The --modelFile is None"
        import pickle
        with open(model_file, 'rb') as ifh_models:
            model_dict_load = pickle.load(ifh_models)
        
        ## ------------ load params for desc calc ------------
        desc_list, ml_model, model_cofig = model_dict_load['desc'], model_dict_load['model'], model_dict_load['config']
        desc_calc_param = model_dict_load['param']['calculator']
        desc_norm_param = model_dict_load['param']['normalization']
        desc_impu_param = model_dict_load['param']['imputation']

        ## determine the descriptor types/calculators needs to be calculated
        desc_calculator_list, custDesc_list = get_desc_descriptors(desc_list, desc_calc_param)

    ## ------------ load data ------------
    print(f">>Step-1: load data ...")
    assert inputType in ['csv', 'sdf'], print(f"\tError, --inputType should be either <csv> or <sdf>")
    if inputType == 'csv':
        mol_dict = load_csv_file(fileNameIn, colName_mid, colName_smi, custDesc_list, colName_expt, sep=sep)
    elif inputType == 'sdf':
        mol_dict = load_sdf_file(fileNameIn, colName_mid, colName_smi, custDesc_list, colName_expt)
        

    ## ------------ prepare descriptors ------------
    print(f">>Step-2: prepare descriptors ...")
    dataTable_desc, dataTable_desc_raw = prep_desc(mol_dict, colName_smi, desc_list, desc_calculator_list, custDesc_list, desc_norm_param, desc_impu_param)

    ## ------------ make predictions ------------
    X = dataTable_desc[desc_list]
    print(f"\tThe shape of X is {X.shape}")
    y_pred = ml_model.predict(X)
    if model_cofig['logy']:
        y_pred = 10**y_pred
    dataTable_desc["Prediction"] = y_pred

    ## merge data
    import pandas as pd
    dataTable_results = pd.merge(left=dataTable_desc[[colName_mid, colName_smi, "Prediction"]], right=dataTable_desc_raw, on=colName_mid, how='outer')
    dataTable_results.to_csv(filePathOut, index=False)
    print(f"\tThe result file has been saved to <{filePathOut}>")

if __name__ == '__main__':
    main()