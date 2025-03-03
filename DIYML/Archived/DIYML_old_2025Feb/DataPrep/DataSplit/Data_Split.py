import warnings
warnings.filterwarnings("ignore")

import os
import sys
import copy
import time
import random
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw

## custom modules
# sys.path.append(os.getcwd())
# import ML_ButinaSplit

## ================================================================================================
## ==================================== random split ==============================================
## ================================================================================================
def nFoldSplit_random(molDict, CV=10, rng=666666, hasVal=True):
    assert CV <= len(molDict), f"\tError, the N_fold ({CV}) is out of range {len(molDict)}"
    list_mol_idx = list(molDict.keys())

    # Shuffle the list using random seed 
    np.random.seed(rng)
    np.random.shuffle(list_mol_idx)

    # Split the list into N sublists
    sublists = np.array_split(list_mol_idx, CV)

    idx_test = sublists[CV-1]
    idx_val = sublists[CV-2] if hasVal else []
    idx_train = [i for i in list_mol_idx if i not in idx_test and i not in idx_val]
    print(f"\t\tSplit the data (n={len(list_mol_idx)}) into Train({len(idx_train)}), Val({len(idx_val)}), and Test({len(idx_test)})")
    # return idx_train, idx_test, idx_val

    # ## update the molDict
    # for idx in molDict:
    #     if idx in idx_test:
    #         molDict[idx]['dataSet'] = 'Test'
    #     elif idx in idx_val:
    #         molDict[idx]['dataSet'] = 'Validation'
    #     else:
    #         molDict[idx]['dataSet'] = 'Training'
    
    return idx_train, idx_val, idx_test

## ================================================================================================
## ==================================== temporal split ============================================
## ================================================================================================
def nFoldSplit_temporal(molDict, dataTable, CV=10, rng=666666, hasVal=True, colName_date="Created On"):
    assert CV <= len(molDict), f"\tError, the N_fold ({CV}) is out of range {len(molDict)}"

    ## check if required columns exist
    if colName_date not in dataTable.columns:
        print(f"\tError! The date column <{colName_date}> does not exist; Using <random> split method instead opf <temporal> method")
        idx_train, idx_val, idx_test = nFoldSplit_random(molDict, CV=CV, rng=rng, hasVal=hasVal)
    else:
        try:
            dataTable["date_formatted"] = pd.to_datetime(dataTable[colName_date])
            dataTable_sorted = dataTable.sort_values(by=["date_formatted"], ascending=[True])
        except Exception as e:
            print(f"\tWarning! The mol date column <{colName_date}> cannot be formatted; Using <random> split method instead opf <temporal> method")
            idx_train, idx_val, idx_test = nFoldSplit_random(molDict, CV=CV, rng=rng, hasVal=hasVal)
        else:
            # Split the list into N sublists
            list_mol_idx = dataTable_sorted.index.to_numpy()
            sublists = np.array_split(list_mol_idx, CV)
            
            idx_test = sublists[CV-1]
            idx_val = sublists[CV-2] if hasVal else []
            idx_train = [i for i in list_mol_idx if i not in idx_test and i not in idx_val]
            print(f"\tSplit the data (n={len(list_mol_idx)}) into Train({len(idx_train)}), Val({len(idx_val)}), and Test({len(idx_test)})")

    return idx_train, idx_val, idx_test

## ================================================================================================
## ====================================== butina split ============================================
## ================================================================================================
def nFoldSplit_butina(molDict, CV=10, rng=666666, hasVal=True):
    assert CV <= len(molDict), f"\tError, the N_fold ({CV}) is out of range {len(molDict)}"

    ## ------------------ split training (including validation) and test ------------------
    ## initiate a splittor
    mySplittor = ButinaSplit(descType='FPs', simCutoff_cluster=0.8, simCutoff_refine=0.85, rng=rng)

    ## fit the splittor with data (list of smiles)
    List_smiles_all = [molDict[idx]['Smiles'] for idx in molDict]
    mySplittor.fit(List_smiles_all)

    ## export results
    print(f"\t\t==> Original split: training (+validation) [{len(mySplittor._idxs_train_raw)}]; test [{len(mySplittor._idxs_test_raw)}]")
    print(f"\t\t==> Removing {len(mySplittor._idxs_train_drop)} training(+validation) mols similar to test mols")
    print(f"\t\t==> Final split: training (+validation) [{len(mySplittor._idxs_train_refined)}]; test: [{len(mySplittor._idxs_test_refined)}]")

    ## save results
    idx_test = mySplittor._idxs_test_refined
    idx_trainVal = mySplittor._idxs_train_refined
    idx_train, idx_val = idx_trainVal, []
     
    ## ------------------ split training and validation ------------------
    if hasVal:
        idx_train, idx_val = [], []
        ## initiate a splittor
        mySplittor_2 = ButinaSplit(descType='FPs', simCutoff_cluster=0.8, simCutoff_refine=0.85, rng=rng)

        ## fit the splittor with data (list of smiles)
        List_smiles_trainVal = [molDict[idx]['Smiles'] for idx in idx_trainVal]
        mySplittor_2.fit(List_smiles_trainVal)

        ## export results
        print(f"\t\t==> Original split: training [{len(mySplittor_2._idxs_train_raw)}]; validation [{len(mySplittor_2._idxs_test_raw)}]")
        print(f"\t\t==> Removing {len(mySplittor_2._idxs_train_drop)} training mols similar to validation mols")
        print(f"\t\t==> Final split: training [{len(mySplittor_2._idxs_train_refined)}]; validation [{len(mySplittor_2._idxs_test_refined)}]")

        ## save results
        idx_val = mySplittor_2._idxs_test_refined
        idx_train = mySplittor_2._idxs_train_refined

    # ## ------------------ update molDict label ------------------
    # for idx in molDict:
    #     if idx in idx_test:
    #         molDict[idx]['dataSet'] = 'Test'
    #     elif idx in idx_val:
    #         molDict[idx]['dataSet'] = 'Validation'
    #     else:
    #         molDict[idx]['dataSet'] = 'Training'

    return idx_train, idx_val, idx_test

######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
'''
This is the class of spliter which dose diverse compound selection based on Butina clustring. 

'''

#########################################################################################
############################### Regression model ########################################
#########################################################################################
class ButinaSplit(object):
    ## <----- model initiation ---->
    def __init__(self, descType='FPs', simCutoff_cluster=0.8, simCutoff_refine=0.8, rng=666666):     
        # assert myScikitModel is None, f"Warning! Please define a model, Got: {myScikitModel}"
        self._descType = descType
        self._simCutoff_cluster = simCutoff_cluster
        self._simCutoff_refine = simCutoff_refine
        self._rng= rng
     
    ## <----- fit object with mol info ---->
    def fit(self, List_mols, calcFPS=True, save=False, rng=666666):
        ## count time
        beginTime = time.time()
        ## ------------------------------------------------------------------------------------------------
        num_mols=len(List_mols)

        ## ===================== collect mol info and molecular fingerprint ===================== 
        if calcFPS:
            List_Descs = calcDescriptors(self._descType, List_mols, molType='Smiles', nBits=2048)
        else:
            List_Descs = List_mols

        ## ===================== 1) do Butina clustering =====================
        dict_clusters, dict_similarMols, List_Descs = _ButinaClustering(List_Descs, self._simCutoff_cluster)

        ## save the data dict into object
        if save:
            self._dict_clusters = dict_clusters
            self._dict_similarMols = dict_similarMols
            if calcFPS:
                self._fps = List_Descs

        ## ===================== 2) splitting train & test based on the clusters =====================
        self._idxs_train_raw, self._idxs_test_raw = _CustomSelectFromCluster(dict_clusters, num_mols, percentage_total=0.1, percentage_cluster=0.9, rng=self._rng)

        ## ===================== 3) remove similar compounds in training set against test set =====================
        if self._simCutoff_refine < 1:
            self._idxs_train_refined, self._idxs_test_refined, self._idxs_train_drop = _rmSimCompound(List_Descs, 
                                                                                                      self._idxs_train_raw, 
                                                                                                      self._idxs_test_raw, 
                                                                                                      self._simCutoff_refine,
                                                                                                      smi=False,
                                                                                                      calcFPS=False)
        ## ------------------------------------------------------------------------------------------------
        print(f'\t', "-------- Split completed --------")
        print(f'\t', f"Data were split into 1) {len(self._idxs_train_refined)}; 2) {len(self._idxs_test_refined)}")
        print(f'\t', "Clustering and split costs time = %ds ................" % (time.time()-beginTime))
        return self

###################################################################################
###################################################################################
## --------------------------------------------------------
## function for Butina clustering, modified from 
## https://github.com/rdkit/rdkit-orig/blob/master/rdkit/ML/Cluster/Butina.py#L68
## --------------------------------------------------------
def _ButinaClustering(List_Descs, simCutoff=0.8, descNorm=False):
    assert simCutoff>0 and simCutoff<1, f"Error! The similarity threshold for clustering should in [0, 1]."
    print(f'\t', f"-------- Now performing butina clustering (similarity cutoff = {simCutoff}) --------")

        
    ## =====================  perform Butina clustering ===================== 
    ## ------------- Step 1. fine neighbor/similar mols -------------
    #### Step 2.1 calculate the similarity and find the similar mols for each mol
    dict_similarMols = FindSimilarMols(List_Descs, algorithm='Tanimoto', simCutoff=simCutoff, descNorm=descNorm)
    #### Step 2.2 Sort by the number of near neighbors (larger to smaller)
    dict_similarMols_sorted = dict(sorted(dict_similarMols.items(), key=lambda x: len(x[1]), reverse=True))

    ## ------------- Step 2. create a list to monitor the access of mols -------------
    seen = [0] * len(List_Descs)    ## 0 means the mol has not been picked and moved into a cluster

    ## ------------- Step 3. loop from the mol with most neighbors -------------
    dict_clusters = {}
    for mol_idx in dict_similarMols_sorted:
        cluster_name = "Neighbors_Of_"+str(mol_idx)
            
        ## check the host mol
        if seen[mol_idx] == 0:
            dict_clusters[cluster_name] = []
            dict_clusters[cluster_name].append(mol_idx)
            seen[mol_idx] = 1
            
        ## check the neighbor mols
        if len(dict_similarMols_sorted[mol_idx]) > 0:
            for idx_neighbor in dict_similarMols_sorted[mol_idx]:
                if seen[idx_neighbor] == 0:
                    if cluster_name not in dict_clusters:
                        dict_clusters[cluster_name] = []
                    dict_clusters[cluster_name].append(idx_neighbor)
                    seen[idx_neighbor] = 1
    # dict_clusters_sorted = dict(sorted(dict_clusters.items(), key=lambda x: len(x[1]), reverse=True))
    print(f'\t', "There are total %d clusters generated" % (len(dict_clusters)))
    return dict_clusters, dict_similarMols, List_Descs

def _ButinaClustering_new(List_Descs, simCutoff=0.8, descNorm=False):
    assert simCutoff>0 and simCutoff<1, f"Error! The similarity threshold for clustering should in [0, 1]."
    print(f"-------- Now performing butina clustering (similarity cutoff = {simCutoff}) --------")

        
    ## =====================  perform Butina clustering ===================== 
    ## ------------- Step 1. fine neighbor/similar mols -------------
    #### Step 2.1 calculate the similarity and find the similar mols for each mol
    dict_similarMols = FindSimilarMols(List_Descs, algorithm='Tanimoto', simCutoff=simCutoff, descNorm=descNorm)
    #### Step 2.2 Sort by the number of near neighbors (larger to smaller)
    dict_similarMols_sorted = dict(sorted(dict_similarMols.items(), key=lambda x: len(x[1]), reverse=True))

    ## ------------- Step 2. create a list to monitor the access of mols -------------
    seen = [0] * len(List_Descs)    ## 0 means the mol has not been picked and moved into a cluster

    ## ------------- Step 3. loop from the mol with most neighbors -------------
    dict_clusters = {}

    # dict_clusters_sorted = dict(sorted(dict_clusters.items(), key=lambda x: len(x[1]), reverse=True))
    print("There are total %d clusters generated" % (len(dict_clusters)))
    return dict_clusters, dict_similarMols, List_Descs

## --------------------------------------------------------
## define function to select mols from clusters
## --------------------------------------------------------
def _CustomSelectFromCluster(dict_clusters, num_mols, percentage_total=0.4, percentage_cluster=0.25, log=False, rng=666666):
    random.seed(rng)
    print(f'\t', "-------- Now performing molecules selection based on Clusters --------")
    ## sort the tuple list by size from small to large
    dict_clusters_sorted = dict(sorted(dict_clusters.items(), key=lambda x: len(x[1]), reverse=False))

    ## Initiate some variables
    idxs_train, idxs_test = [], []
    List_clusters_remining = []
    sum_mols_looped = 0
    Num_1cmpd_cluster = 0

    ## looping the cluster and select mols
    print(f'\t', "Start Looping %d clusters:" % (len(dict_clusters_sorted)))
    for cluster_name in dict_clusters_sorted:
        ## get a copy of the mol list for this cluster for next operation
        cluster_list = copy.deepcopy(dict_clusters_sorted[cluster_name])
        ## compare the sum of mols in looped clusters, if not meet the criteria, continue
        
        percentage_mols = percentage_total * percentage_cluster
        # percentage_mols = 0.1
        if len(idxs_test) < num_mols * percentage_mols:
            ## calculate num of mols to be select in this cluster
            if len(cluster_list) > 1:
                print(f'\t', f"The num cmpds in this cluster is {len(cluster_list)}")
                num_mols_2pick = int(np.ceil(len(cluster_list) * percentage_cluster))
                ## start randomly pick mols   
                for i in range(num_mols_2pick):
                    ## generate random index from the cluster list
                    randomPick = random.choice(range(len(cluster_list)))
                    idx_picked = cluster_list[randomPick]
                    #print("    ====> %d" % (idx_picked))
                    idxs_test.append(idx_picked)    ## add index into the selected list
                    cluster_list.pop(randomPick)    ## remove selected index from cluster   

            elif len(cluster_list) == 1:
                num_mols_2pick = 1
                idxs_test.append(cluster_list[0])
                cluster_list.pop(0)
                Num_1cmpd_cluster += 1
            else:
                print("Error! No cmpds in this cluster: %s" % (cluster_name))
                
        ## add the remining cluster back to the List of cluster 
        if len(cluster_list) > 0:
            idxs_train += cluster_list
            # for idx_train in cluster_list:
            #     if idx_train not in idxs_train:
            #         idxs_train.append(idx_train)
            # List_clusters_remining.append(cluster_list)
                
        ## Report progress (only for large clusters)
        if log:
            print(f'\t', " ==>Now there are %d mols (total %d) have been looped totally." % (sum_mols_looped, num_mols))
            ## update the sum of mols in looped clusters
            sum_mols_looped += len(cluster_list)
        else:
            pass

    # ## get train indexes
    # for cluster_list in List_clusters_remining:
    #     idxs_train.extend(cluster_list)

    print("%s Train: %d (%f), Test: %d (%f)" % (f'\t', len(idxs_train), len(idxs_train)/num_mols, len(idxs_test), len(idxs_test)/num_mols))
    return idxs_train, idxs_test

## --------------------------------------------------------
## define function to split data set into train/validation/test and return indexes
## --------------------------------------------------------    
def _rmSimCompound(List_mol, idxs_train, idxs_test, simCutoff=0.8, smi=True, calcFPS=True):
    print(f'\t', "-------- Now performing similar molecules elimination --------")
    assert simCutoff > 0, f"Error! The similarity threshold for removing compounds should > 1."
    ## get mol information and calculate ECFP for training compounds
    Dict_FPs_train = {}
    for idx_train in idxs_train:
        if smi:
            this_mol_train = Chem.MolFromSmiles(List_mol[idx_train])
        else:
            this_mol_train = List_mol[idx_train]
        if calcFPS:
            this_ecfp_train = AllChem.GetMorganFingerprintAsBitVect(this_mol_train, 2, nBits=1024)
        else:
            this_ecfp_train = this_mol_train
        Dict_FPs_train[idx_train] = (this_ecfp_train)

    ## loop the test set to do similarity calculation
    idxs_train_drop = []

    for idx_test in idxs_test:
        ## get mol information and calculate ECFP
        if smi:
            this_mol_test = Chem.MolFromSmiles(List_mol[idx_test])
        else:
            this_mol_test = List_mol[idx_test]
        if calcFPS:
            this_ecfp_test = AllChem.GetMorganFingerprintAsBitVect(this_mol_test, 2, nBits=1024)
        else:
            this_ecfp_test = this_mol_test

        for idx_train in Dict_FPs_train:
            if idx_train not in idxs_train_drop:
                ## get fp and calculate the similarity
                this_ecfp_train = Dict_FPs_train[idx_train]
                this_similarity = DataStructs.TanimotoSimilarity(this_ecfp_train, this_ecfp_test)
                if this_similarity >= simCutoff:
                    idxs_train_drop.append(idx_train)

    ## collect training compounds not removed
    idxs_train_keep = []
    idxs_test_new = []

    for i_text in idxs_test:
        if i_text not in idxs_test_new:
            idxs_test_new.append(i_text)

    for i_drop in idxs_train:
        if i_drop in idxs_train_drop:
            idxs_test_new.append(i_drop)
        else:
            idxs_train_keep.append(i_drop)

    print(f'\t', "Total num of training compds: %d" % (len(idxs_train)))
    print(f'\t', "==> Similarity Cutoff = %.2f" % (simCutoff))
    print(f'\t', "==> Dropped similar compds to test set: %d" % (len(idxs_train_drop)))
    return idxs_train_keep, idxs_test_new, idxs_train_drop


'''
## get the index for train, validation, test set
idxs_trainval, idxs_test, Dict_clusters, dict_similarMols = DiverseSelection.ButinaSplitter_FPs(List_smiles, simCutoff=0.75)

## remove similar compounds in training set against test set
idxs_trainval_keep, idx_test_add = DiverseSelection.rmSimCompound(dataTable_AlphalogD, idxs_trainval, idxs_test, col_Smile='Smiles', simCutoff=0.9)

## save data
DataTable_trainval = dataTable_AlphalogD.iloc[idxs_trainval_keep]
DataTable_trainval_merged = dataTable_mergedLogD.iloc[idxs_trainval_keep+idxs_ADMElogD]
# DataTable_trainval_merged.to_csv(os.path.join(WorkDir, 'data_merged_logD_trainval_%d_%s.csv'%(len(DataTable_trainval_merged), dateToday)), index=False)
print(DataTable_trainval.shape, DataTable_trainval_merged.shape)

DataTable_test = dataTable_AlphalogD.iloc[idxs_test+idx_test_add]
# DataTable_test.to_csv(os.path.join(WorkDir, 'data_merged_logD_test_%d_%s.csv'%(len(DataTable_test), dateToday)), index=False)
print(DataTable_test.shape)




## save data
DataTable_trainval = dataTable_AlphalogD.iloc[idxs_trainval_keep]
# DataTable_trainval.to_csv(os.path.join(WorkDir, 'data_Alpha_logD_trainval_%d_%s.csv'%(len(DataTable_trainval), dateToday)), index=False)
print(DataTable_trainval.shape)

DataTable_test = dataTable_AlphalogD.iloc[idxs_test+idx_test_add]
# DataTable_test.to_csv(os.path.join(WorkDir, 'data_Alpha_logD_test_%d_%s.csv'%(len(DataTable_test), dateToday)), index=False)
print(DataTable_test.shape)
'''


######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################

## -----------------------------------------------------------------------------------
'''
This is a collection of tools will be used in ML_ButinaSplit.py module

'''

## -----------------------------------------------------------------------------------
# https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html

def calcDescriptors(descType, List_mols, molType='Smiles', nBits=1024, getBi=False):
    List_Descs = []
    for this_mol in List_mols:
        ## get mol from Smiles
        if molType == 'Smiles':
            this_mol = Chem.MolFromSmiles(this_mol)

        thisDesc, bi = None, {} 
        
        ## ## get descriptors/FPs from mol
        if descType == 'FPs':
            thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=1, nBits=nBits)
        elif descType == 'ECFP2':
            thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=1, nBits=nBits)
        elif descType == 'ECFP4':
            thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=2, bitinfo=bi)
        elif descType == 'ECFP6':
            thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=3, nBits=nBits)        
        else:
            # thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=2, nBits=nBits, bitinfo=bi)
            thisDesc = AllChem.GetMorganFingerprintAsBitVect(this_mol, radius=2, nBits=nBits)
        
        ## collect the calculated descriptprs and store in list
        if getBi:
            List_Descs.append((thisDesc, bi))
        else:
            List_Descs.append(thisDesc)
    return List_Descs


## -----------------------------------------------------------------------------------
## normalize/scale the descriptors
def _descNormalization(List_Descs, descNorm):
    print(f"The scale method is: {descNorm}")
    List_Descs = copy.deepcopy(List_Descs)
    return List_Descs

## -----------------------------------------------------------------------------------
## calculate the similarity between a and b based on specified algorithm
def _calcSimilarity(a, b, algorithm='Tanimoto'):
    if algorithm == 'Tanimoto':
        similarity = DataStructs.TanimotoSimilarity(a, b)
    else:
        similarity = DataStructs.TanimotoSimilarity(a, b)
    return similarity

## -----------------------------------------------------------------------------------
def FindSimilarMols(List_Descs, algorithm='Tanimoto', simCutoff=0.8, descNorm=None):
    if descNorm:
        List_Descs = _descNormalization(List_Descs, descNorm)

    num_mols = len(List_Descs)
    ## create empty dict and fill the dict by similar compounds
    dict_similarMols = {}
    for i in range(num_mols):
        dict_similarMols[i] = []
        for j in range(i):
            similarity_ij = _calcSimilarity(List_Descs[i], List_Descs[j], algorithm=algorithm)
        
            ## if mol[i] and mol[j] is similar enough, add into matrix
            if similarity_ij >= simCutoff:
                dict_similarMols[i].append(j)
                dict_similarMols[j].append(i)
    return dict_similarMols


## -----------------------------------------------------------------------------------

def VisualizeMolFPs(mol, fp, bitInfo, bitList=None, num_mol_shown=10, molsPerRow=5):
    if not bitList:
        bitList = fp.GetOnBits()
    tpls = [(mol, x, bitInfo) for x in bitList]
    legends = [str(x) for x in bitList][:num_mol_shown]
    visual = Draw.DrawMorganBits(tpls[:num_mol_shown], molsPerRow=molsPerRow, legends=legends)
    return visual


## -----------------------------------------------------------------------------------

def CalcFPsFromTable(dataTable, colName_smi='Smiles', FPs_type='ECFP6', nBits=1024, getBi=False):

    dataTable_new = dataTable.reset_index(names='index_original')

    dataList_fps = []
    dataDict_bitInfo = {}
    for idx in dataTable_new.index:

        if idx % 5000 == 0:
            print(idx)

        if dataTable_new[colName_smi].notna()[idx]:
            ## calculate the FPs and bitInfo
            smi = dataTable_new[colName_smi][idx]
            try:
                List_Descs = calcDescriptors(descType=FPs_type, List_mols=[smi], nBits=nBits, getBi=getBi)
            except Exception as e:
                print(f'Cannot calc FPs for mol {idx}. Msg: {e}')
                fps, bi = None, {}
            else:
                if getBi:
                    fps, bi = List_Descs[0][0], List_Descs[0][1]
                else:
                    fps, bi = List_Descs[0], {}
        else:
            fps, bi = None, {}        
        # Convert the fingerprints into a binary string
        # fp_list = list(map(int, fp.ToBitString()))
        fps_array = np.array(fps)

        ## collect the FPs and bitInfo   
        dataList_fps.append(fps_array)

        index_ori = dataTable_new['index_original'][idx]
        dataDict_bitInfo[index_ori] = bi
    
    df_fps = pd.DataFrame(dataList_fps, index=dataTable_new.index, columns=[f'FP_bit_{i}' for i in range(nBits)])
    df_merged = dataTable_new.merge(df_fps, left_index=True, right_index=True).set_index('index_original')
    df_merged.index.name = None
    return df_merged, dataDict_bitInfo

## -----------------------------------------------------------------------------------

######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################