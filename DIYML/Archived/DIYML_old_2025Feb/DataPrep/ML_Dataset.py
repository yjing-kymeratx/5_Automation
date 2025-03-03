##############################################################################################
###################################### import packages #######################################
##############################################################################################
## avoid python warning if you are using > Python 3.11, using action="ignore"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

## load packages
import os
import time
import chardet
import numpy as np
import pandas as pd

from rdkit import Chem
from DataPrep.Descriptors import Desc_ChemAxon, Desc_MolFPs, Desc_RDKit
from DataPrep.DataSplit import Data_Split

##############################################################################################
################################# tools for processing data ##################################
##############################################################################################
## ============ tools for loading data ============
def _determine_encoding(fileNameIn, default='utf-8'):
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

def _cleanUpSmiles(smi):
    try:
        ## SMILES Notation cleaner
        smi = smi.split("|")[0] if "|" in smi else smi
        ## tbd
        smi = smi.replace("\n", "").replace("\r", "").replace("\r\n", "")
        ##
        if "\\" in smi:
            smi = smi.replace('\\', '\\\\')
            print(f'\tThere is a "\\" in the SMILES. Adding 1 "\\" into the SMILES, now new SMILES is {smi}')
        ## strp salts
        if '.' in smi:
            smi = max(smi.split('.'), key=len)
        ## SMILES Canonicalization
        rdMol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(rdMol)
    except Exception as e:
        print(f"\tCannot clean the smi <{smi}>; Errow msg: {e}")
        smi = np.nan
    return smi

##############################################################################################
######################### data class for machine learning modeling ###########################
##############################################################################################
class Data4ML(object):
    ## ================================================ initialize object ================================================
    def __init__(self, dataName="myData"):
        self._name = dataName
        self._dataTableRaw = None
        self._molDict = None

    ## ================================================ load data from CSV ================================================
    def load_csv(self, fileNameIn, sep=",", colName_mid="Compound Name", colName_smi="Smiles", colName_activity="AssayData", istrain=True):
        ## check file
        print(f"1. Loading csv from {fileNameIn}")
        assert os.path.exists(fileNameIn), f"\tFile {fileNameIn} does not exist"
        self.setAttributes("_fileNameIn", fileNameIn)

        ## check encoding
        encoding = _determine_encoding(fileNameIn)

        ## load data from csv
        try:
            ## read csv file
            dataTable = pd.read_csv(fileNameIn, sep=sep, encoding=encoding)
            print(f"\tThe original csv file has {dataTable.shape[0]} rows and {dataTable.shape[1]} columns")              
        except Exception as e:
            print(f"\tCannot read cvs file, error: {e}")
        else:
            ## check if required columns exist
            assert colName_mid in dataTable.columns, f"Error! The mol ID column <{colName_mid}> does not exist"
            self.setAttributes("_colNameID", colName_mid)
            print(f"\tColumn for compound ID is {colName_mid}")

            assert colName_smi in dataTable.columns, f"Error! The mol SMILES column <{colName_smi}> does not exist"
            self.setAttributes("_colNameSmi", colName_smi)
            print(f"\tColumn for compound SMILES is {colName_smi}")

            ## clean smi
            dataTable[colName_smi+'_raw'] = dataTable[colName_smi]
            dataTable[colName_smi] = dataTable[colName_smi].apply(lambda smi: _cleanUpSmiles(smi))

            if istrain:
                assert colName_activity in dataTable.columns, f"Error! The activity column <{colName_activity}> does not exist"
                self.setAttributes("_colNameActivity", colName_activity)
                print(f"\tColumn for compound activity is {colName_activity}")
                key_cols = [colName_mid, colName_smi, colName_activity]
            else:
                print(f"\tNo Column for compound activity in prediction set")
                key_cols = [colName_mid, colName_smi]

            ## remove the rows with nan
            dataTable_new = dataTable.dropna(subset=key_cols, how='any').reset_index(drop=True)
            print(f"\tThe cleaned csv file has {dataTable_new.shape[0]} rows and {dataTable_new.shape[1]} columns")
            
            ## save data
            # self.setAttributes("_dataTableRaw", dataTable)
            self.setAttributes("_dataTable", dataTable_new)

            ## get mol dictionary
            molDict = self.__extract_mol_data()
            self.setAttributes("_molDict", molDict)
        return None
        
    def load_sdf(self, fileNameIn, colName_mid="Compound Name", colName_smi="Smiles", colName_activity="AssayData", istrain=False):
        ## to be added
        try:
            mols = [mol for mol in Chem.SDMolSupplier(fileNameIn)]
        except Exception as e:
            print(f"\tCannot load the sdf file {fileNameIn}. Error msg: {e}")
        else:
            molDict = {}
            for idx in range(len(mols)):
                ## --------- mol smi ---------
                try:
                    smi = tool_mol_2_smi(mols[idx])
                except:
                    smi = np.nan
                    print(f"\tCannot parse this {idx} mol {mols[idx]}")
                else:
                    molDict[idx] = {}
                    molDict[idx]['idx'] = idx
                    molDict[idx]['rdMol'] = mols[idx]
                    molDict[idx]["Smiles"] = smi
                    molDict[idx][colName_smi] = smi
                    ## --------- mol id ---------
                    try:
                        mid = mols[idx].GetProp(colName_cid)
                    except:
                        mid = f"Molecule_{idx}"
                    molDict[idx]["molID"] = mid
                    molDict[idx][colName_cid] = mid

        self.setAttributes("_molDict", molDict)
        print(f"\tTotal {len(molDict)} mols were extracted.")
        return None
        
    ## ================================================ calculate descriptors ================================================
    def calc_desc(self, desc_fps=True, desc_rdkit=True, desc_cx=True):
        print(f"2. Calculating descriptors (FPs {desc_fps}; ChemAxon {desc_rdkit}; RDKit: {desc_cx}) ... ")
        assert self._molDict is not None, f"\tError, self._molDict is None, pls check the data loading from csv."
        molDict = self._molDict

        ## ------------ calculate chemAxon properties ------------
        self.setAttributes("_desc_cx", desc_cx)
        if desc_cx:
            rmProps = ['polar-surface-area_unit', 'pka_apKa1', 'pka_apKa2', 'pka_bpKa1', 'pka_bpKa2']
            self.setAttributes("_desc_cx_param", {"ip": '172.31.19.252', "port": '8064', "calculator": 'calculate', "rmProps": rmProps})
            molDict = Desc_ChemAxon.calc_desc_chemaxon(molDict = molDict, 
                                                        ip=self._desc_cx_param["ip"], 
                                                        port=self._desc_cx_param["port"], 
                                                        calculator=self._desc_cx_param["calculator"],
                                                        rmProps=self._desc_cx_param["rmProps"])
        ## ------------ calculate mol fingerprints ------------
        self.setAttributes("_desc_fps", desc_fps)
        if desc_fps:
            self.setAttributes("_desc_fp_param", {"fpType": "ECFP", "radius": 3, "nBits": 2048})
            molDict = Desc_MolFPs.calc_desc_fingerprints(molDict=molDict, 
                                                        fpType=self._desc_fp_param["fpType"], 
                                                        radius=self._desc_fp_param["radius"], 
                                                        nBits=self._desc_fp_param["nBits"])
        ## ------------ calculate rdkit properties ------------
        self.setAttributes("_desc_rdkit", desc_rdkit)
        if desc_rdkit:
            self.setAttributes("_desc_rdkit_param", {"physChem": True, "subStr": True, "clean": True})
            molDict = Desc_RDKit.calc_desc_rdkit(molDict=molDict, 
                                                physChem=self._desc_rdkit_param["physChem"], 
                                                subStr=self._desc_rdkit_param["subStr"], 
                                                clean=self._desc_rdkit_param["clean"])
        ## ------------ update the molDict ------------
        self.setAttributes("_molDict", molDict)

    ## ================================================ data split ================================================
    ## separate training/test/validation set
    def train_val_test_split(self, split_method='random', CV=10, hasVal=True, rng=666666, colName_date="Created On"):
        print(f"3. Data split using {split_method} method ... ")
        assert self._molDict is not None, f"\tError, self._molDict is None, pls check the data loading from csv."
        ##
        if split_method not in ['random', 'temporal', 'butina']:
            print(f"\tError, split method should be in the list ['random', 'temporal', 'butina']")
            split_method == 'random'
        self.setAttributes("_split_method", split_method)

        ## ======================= split =======================
        molDict = self._molDict
        dataTable = self._dataTable

        if split_method == 'random':
            idx_train, idx_val, idx_test = Data_Split.nFoldSplit_random(molDict, CV=CV, rng=rng, hasVal=hasVal) 

        elif split_method == 'temporal':
            self.setAttributes("_colNameDate", colName_date)
            print(f"\tColumn for compound date is <{colName_date}>")
            idx_train, idx_val, idx_test = Data_Split.nFoldSplit_temporal(molDict, dataTable, CV=CV, rng=rng, hasVal=True, colName_date=colName_date)
    
        elif split_method == 'butina':
            idx_train, idx_val, idx_test = Data_Split.nFoldSplit_butina(molDict, CV=CV, rng=rng, hasVal=hasVal)
        ## update
        self.setAttributes("_idx_train", idx_train)
        self.setAttributes("_idx_val", idx_val)
        self.setAttributes("_idx_test", idx_test)  
        ## 
        for idx in molDict:
            if idx in idx_test:
                molDict[idx]['dataSet'] = 'Test'
            elif idx in idx_val:
                molDict[idx]['dataSet'] = 'Validation'
            else:
                molDict[idx]['dataSet'] = 'Training'
        self.setAttributes("_molDict", molDict)
        
    ## ================================================ data split ================================================
    def prepare_dataset(self, dsLabels=['Training', 'Validation', 'Test'], istrain=True):
        print(f"4. Preparing Dataset ... ")
        ## initiate the dict for all
        dict_AllDataSets = {}
        for dsLabel in dsLabels:
            dict_AllDataSets[dsLabel] = {}
            dict_AllDataSets[dsLabel]['X'] = {}
            if istrain:
                dict_AllDataSets[dsLabel]['y'] = {}
        
        ## loop the molDict and update the dictionary    
        for idx in self._molDict:
            dsLabel = self._molDict[idx]['dataSet']
            assert dsLabel in dict_AllDataSets, f"Error! The {idx} dataset label {dsLabel} is not in your label list {dsLabels}"
            ##
            dict_AllDataSets[dsLabel]['X'] = self.__updatingDescDict(dict_AllDataSets[dsLabel]['X'], idx)
            if istrain:
                dict_AllDataSets[dsLabel]['y'][idx] = {}
                dict_AllDataSets[dsLabel]['y'][idx][self._colNameActivity] = self._dataTable[self._colNameActivity][idx]

        ## updating the object
        for dsLabel in dict_AllDataSets:
            for vLabel in dict_AllDataSets[dsLabel]:
                thisDict = dict_AllDataSets[dsLabel][vLabel]
                thisTable = pd.DataFrame.from_dict(thisDict).T
                thisLabel = f"{vLabel}_{dsLabel}"
                self.setAttributes(thisLabel, thisTable)
        print(f"\tPrepared Dataset (pandas dataframe) has been saved to <X_Training, y_Training, X_Validation, y_Validation, X_Test, y_Test>")

    ## ================================================
    ## ========= some internal tools of basic =========
    ## ================================================
    ## set attributes with values outside of __init__
    def setAttributes(self, attrName, attrValue):
        setattr(self, attrName, attrValue)

    def __extract_mol_data(self):
        dataTable = self._dataTable
        colName_cid = self._colNameID
        colName_smi = self._colNameSmi
        dataDict_mol = {}

        ## loop the data and extract mol info
        for idx in dataTable.index:
            ## check smiles structure
            if dataTable[colName_smi].notna()[idx]:
                cid = dataTable[colName_cid][idx]
                smi = dataTable[colName_smi][idx]
                rdMol = Chem.MolFromSmiles(smi)

                if idx not in dataDict_mol:
                    dataDict_mol[idx] = {}
                    dataDict_mol[idx]['idx'] = idx
                    dataDict_mol[idx][colName_cid] = cid
                    dataDict_mol[idx][colName_smi] = smi
                    dataDict_mol[idx]['molID'] = cid
                    dataDict_mol[idx]['Smiles'] = smi
                    dataDict_mol[idx]['rdMol'] = rdMol
        print(f"\tTotal {len(dataDict_mol)} mols were extracted.")
        return dataDict_mol
        
    ## ======================================================
    ## ========= some tools of preparing descriptors =========
    ## ======================================================
    def __updatingDescDict(self, myDict, idx):
        ## define private function to update subdict
        def __updateSubDict(dict_all, dict_sele, rmItem=[], prefix_new=''):
            for item in dict_all:
                if item not in rmItem:
                    item_new = prefix_new + item
                    if item_new not in dict_sele:
                        dict_sele[item_new] = dict_all[item]
            return dict_sele

        ## molecular fingerprints
        this_Mol = self._molDict[idx]
        myDict[idx] = {}
        if self._desc_fps:
            if 'desc_fps' in this_Mol:
                this_desc = this_Mol['desc_fps']
                myDict[idx] = __updateSubDict(this_desc, myDict[idx], rmItem=['Fps'])
            else:
                print(f"Error, <desc_fps> is not prepared, please check self._molDict")
        
        ## rdkit molecular properties
        if self._desc_rdkit:
            if 'desc_rdkit' in this_Mol:
                this_desc = this_Mol['desc_rdkit']
                myDict[idx] = __updateSubDict(this_desc, myDict[idx], rmItem=[], prefix_new='rd_')
            else:
                print(f"Error, <desc_rdkit> is not prepared, please check self._molDict")

        ## chemAxon molecular properties
        if self._desc_cx:
            if 'desc_cx' in this_Mol:
                this_desc = this_Mol['desc_cx']
                myDict[idx] = __updateSubDict(this_desc, myDict[idx], rmItem=[])
            else:
                print(f"Error, <desc_cx> is not prepared, please check self._molDict")
        
        ## export
        return myDict
    
    ## ======================================================





