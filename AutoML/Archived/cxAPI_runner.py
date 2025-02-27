#!/usr/bin/env python3

## =====================================================================================   
## =================================== load packages ===================================
## =====================================================================================
# Optional step: Remove warnings thrown by invalid SSL certificate.
import warnings
warnings.filterwarnings('ignore')

import os
import time
import datetime
import pickle
import shutil
import argparse

import ast
import subprocess

import numpy as np
import pandas as pd

## custom module
from myTools import D360_API

## =====================================================================================   
## =================================== load argparses ==================================
## =====================================================================================
def Step_0_load_args():
    parser = argparse.ArgumentParser(description='Usage ot cxcalc_runner.py, details to be added later')
    parser.add_argument('-q', action="store", type=int, default=None, help='D360 Query ID')
    parser.add_argument('-i', action="store", default=None, help='The input file downloaded from D360')
    parser.add_argument('--colName_cid', action="store", default="Compound Name", help='The column name of mol KT ID')
    parser.add_argument('--colName_bid', action="store", default="Concat;Batch Name", help='The column name of batch ID')
    parser.add_argument('--colName_smi', action="store", default="Structure", help='The column name of SMILES')
    parser.add_argument('--colName_prj', action="store", default="Concat;Project", help='The column name of Projects')

    args = parser.parse_args()

    return args

## =====================================================================================   
## ======================= load data table for cxcalc calculation ======================
## =====================================================================================
def folderChecker(my_folder='my_folder'):
    # Check if the folder exists
    check_folder = os.path.isdir(my_folder)
    # If the folder does not exist, create it
    if not check_folder:
        os.makedirs(my_folder)
        print(f"\tCreated folder:", my_folder)
    else:
        pass
    return my_folder

def Step_1_load_data_table(args):
    beginTime = time.time()

    ## --------------------- load data table ------------------------------------
    if args.q is not None:    
        user_name="yjing@kymeratx.com"
        tokenFile = './myTools/yjing_D360.token'        
        dataTableFile_Raw = D360_API.dataDownload(my_query_id=args.q, user_name=user_name, tokenFile=tokenFile)
    elif args.i is not None:
        dataTableFile_Raw = args.i
    else:
        print(f"\tError! Both <-q> (Query ID) and <-i> (input file) are None.")    ## alert
        exit()

    ## read data table
    dataTable = pd.read_csv(dataTableFile_Raw).reset_index(drop=True)
    print(f"\tThe original data table has shape {dataTable.shape}")
    assert dataTable.shape[0] > 0, f"Script Terminated! There is no molecule downloaded with Smiles/structure!"    ## alert

    ## move the raw file to tmp folder
    dataTableFileName = dataTableFile_Raw.split('/')[0]

    tmp_folder = folderChecker(f"./tmp")
    shutil.move(dataTableFile_Raw, f"{tmp_folder}/{dataTableFileName}")
    print(f"\tMove the downloaded file {dataTableFile_Raw} to ./tmp/{dataTableFileName}")

    ## -------------------- clean up table -----------------------------------
    colName_cid = args.colName_cid
    colName_bid = args.colName_bid
    colName_smi = args.colName_smi
    colName_prj = args.colName_prj

    try:
        dataTable = dataTable.dropna(subset=[colName_cid, colName_bid, colName_smi])
        print(f"\tThe non-nan data table has shape {dataTable.shape}")
    except Exception as e:
        print(f"\tWarning! Error in <dropna>. Error msg: {e}")    ## alert
        dataTable = dataTable

    ## ---------------------- clean batch id ------------------------------------
    def get_basic_bid(row, colName_cid="Compound Name", colName_bid="Concat;Batch Name"):
        try:
            cid, bid = row[colName_cid], row[colName_bid]
            bid_list = sorted(str(bid).split(';'), reverse=False)
            bid_num = bid_list[0].zfill(3)
            bid_new = f"{cid}-{bid_num}"
        except Exception as e:
            print(e)
            bid_new = 'TBD'
        return bid_new

    colName_bic_new = "Molecule-Batch ID"
    dataTable[colName_bic_new] = dataTable.apply(lambda row: get_basic_bid(row, colName_cid, colName_bid), axis=1)

    ## -------------------------------------------------------------------------
    print(f"\tTime to download and/or load and clean data: {round(time.time()-beginTime)} s")
    return dataTable

################################################################################################
################################## run JChem API ###############################################
################################################################################################
## ---------------- tool to generate <dataJson> for API call ----------------
def tool_generate_dataJson(smi, prop_list=None, version='V22'):
    ## predefine the dataJson
    calculations = {}
    calculations["elemental-analysis"] = '{"countAtoms": [1, 6, 8], "countIsotopes": [{"atomNumber": 6, "isotopeNumber": 12}], "operations": "mass, formula", "symbolID": true}'
    calculations["logp"] = '{"atomIncrements": true, "method": "CHEMAXON"}'
    calculations["logd"] = '{"phList": [1.5, 5, 6.5, 7.4]}'    # 'logd': '{"phList": [7.4]}'
    calculations["charge"] = '{"ph": 7.4}'
    calculations["solubility"] = '{"phSequence": {"pHLower": 1.5, "pHStep": 0.1, "pHUpper": 7.4}, "unit": "LOGS"}'

    if version == 'V23':
        calculations["pka"] = '{"micro": false, "outputFormat": "mrv", "outputStructureIncluded": false, "pKaLowerLimit": -20, "pKaUpperLimit": 10, "prefix": "DYNAMIC", "temperature": 298, "types": "pKa, acidic, basic"}'
        calculations['bbb'] = '{}'
        calculations['cns-mpo'] = '{}'
        calculations['polar-surface-area'] = '{"excludePhosphorus": true, "excludeSulfur": true, "outputFormat": "mrv", "outputStructureIncluded": false, "pH": 7.4}'
        calculations["hbda"] = '{"excludeHalogens": true, "excludeSulfur": true, "outputFormat": "mrv", "outputStructureIncluded": false, "pH": 7.4}'
        calculations["topology-analyser"] = '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "ringSize": 0, "ringSystemSize": 0, "operations": "myOperationText", "outputFormat": "mrv"}'
        myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize, ringSystemCount'
    else:
        calculations["pka"] = '{"micro": false, "pKaLowerLimit": -10, "pKaUpperLimit": 20, "prefix": "STATIC", "temperature": 298, "types": "pKa, acidic, basic"}'
        calculations["polar-surface-area"] = '{"excludePhosphorus": true, "excludeSulfur": true, "pH": null}'
        calculations["hbda"] = '{"excludeHalogens": true, "excludeSulfur": true, "resultMoleculeFormat": "mrv", "resultMoleculeIncluded": false, "pH": 7.4}'
        calculations["topology-analyser"] = '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "ringSize": 0, "ringSystemSize": 0, "operations": "myOperationText"}'
        # myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize'
        myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, chiralCenterCount, aromaticRingCount, heteroRingCount'
    calculations["topology-analyser"] = calculations['topology-analyser'].replace('myOperationText', myOperationTopology)

    ## based on the query calculation, prepare the calculators (string)
    prop_list = list(calculations.keys()) if prop_list is None else prop_list

    dataList_calculators = []
    for prop in prop_list:
        if prop in calculations:
            prop_param = calculations[prop]
            dataList_calculators.append(f'"{prop}": {prop_param}')
    api_calculator = ', '.join(dataList_calculators)
    
    ## prepare the dataJson string for API calls
    dataJson = '{"calculations": {%s}, "inputFormat": "smiles", "structure": "%s"}' % (api_calculator, smi)
    return dataJson

def calc_mol_ChemAxon_property(smi, prop_list, version='V22', dataDict_param={'URL_web':'calculator/calculate', 'header':['accept: */*', 'Content-Type: application/json']}, detailedInfo=False):
    ## ----------------------------------------------------------------
    ## define the framework of the cmd
    ip_address = '172.31.25.202' if version == 'V23' else '172.31.25.202'
    URL_web = f'http://{ip_address}:8064/rest-v1/' + dataDict_param['URL_web']    #'http://172.31.19.252:8064/rest-v1/calculator/calculate'
    header1, header2 = dataDict_param['header'][0], dataDict_param['header'][1]   

    ## perpare dataJson for the ChemAxon API calculation
    dataJson = tool_generate_dataJson(smi, prop_list, version=version)
        
    ## Define the command you want to execute
    commandLine = ['curl', '-X', 'POST', URL_web, '-H', header1, '-H', header2, '-d', str(dataJson)]

    ## run the cmd using subprocess package to execute the command
    try:
        process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
        output, error = process.communicate()
        output_decoded = ast.literal_eval(output.decode())
    except Exception as e:
        print(f"Error! Cannot calculate the ChemAxon properties. Error msg: {e}")
        output, output_decoded, error = None, {}, 'Error, Can not generate SMILES from mol!'

    ## ----------------------------------------------------------------
    ## export the results
    
    dataDict_out = {}
    dataDict_out['Smiles_in'] = smi
    dataDict_out['output_raw'] = output_decoded

    if "error" in output_decoded:
        dataDict_out["error"] = output_decoded["error"]
    else:
        ## loop all the results in the output and extract them one by one
        for propType in output_decoded:
            ## eliminate un-wanted properties
            if propType in ["isoelectric-point", "pka-distribution", "major-microspecies"]:
                pass
            else:
                ## ---------------- Error for the calculation ----------------
                if "error" in output_decoded[propType]:
                    dataDict_out[propType] = output_decoded[propType]["error"]["message"]

                ## ---------------- pKa ----------------
                elif propType == 'pka':
                    ## all pKa value list and atom info
                    # if 'structure' in output_decoded['pka']:
                    #     dataDict_out[f'pka_structure'] = output_decoded['structure']
                    if "pkaValuesByAtom" in output_decoded['pka']:
                        dataDict_out[f'pka_list'] = output_decoded['pka']   ## the whole pKa prediction (list)

                    ## acid pKa
                    dataDict_out['pka_apKa1'], dataDict_out['pka_apKa2'] = None, None
                        
                    ## all pKa value list and atom info
                    for pKa_type in ['acidic', 'basic']:
                        if f'{pKa_type}ValuesByAtom' in output_decoded[propType]:
                            reverse = True if pKa_type == 'basic' else False
                            pKa_spec = sorted(output_decoded[propType][f'{pKa_type}ValuesByAtom'], key=lambda x: x['value'], reverse=reverse)
                            for i_pka in range(min([len(pKa_spec), 2])):
                                pKa_colName = [f'{pKa_type[0]}pKa1', f'{pKa_type[0]}pKa2'][i_pka]
                                dataDict_out[f'pka_{pKa_colName}'] = pKa_spec[i_pka]['value']
                
                ## ---------------- logd ----------------
                elif propType == 'logd':
                    for item in output_decoded[propType]['logDByPh']:
                        pH = item['pH']
                        dataDict_out[f'{propType}[pH={pH}]'] = item['value']
                
                ## ---------------- solubility ----------------
                elif propType == 'solubility':
                    unit_solubility = 'logS'
                    for item in output_decoded[propType]['phDependentSolubilities']:
                        dataDict_out[f'{propType}({unit_solubility})'] = item['value']
                
                elif propType == 'solubility':
                    if 'unit' in output_decoded[propType]:
                        unit_solubility = output_decoded[propType]['unit']
                        propType_specific = f"{propType}({unit_solubility}"
                    else:
                        propType_specific = propType
                    for item in output_decoded[propType]:
                        if item == 'intrinsicSolubility':
                            dataDict_out[item] = output_decoded[propType][item]
                        elif item == 'phDependentSolubilities':
                            for sol in output_decoded[propType][item]:
                                this_ph, this_value = sol['pH'], sol['value']
                                dataDict_out[f'{propType_specific}[pH={this_ph}]'] = this_value
                ## ---------------- Other ----------------
                else:
                    for propName in output_decoded[propType]:
                        ## for the atom level detailed information, by pass if <detailedInfo> flag false
                        if not detailedInfo and 'ByAtom' in propName:
                            pass
                        ## cns/bbb score with components
                        elif propType in ['cns-mpo', 'bbb'] and propName =='properties':
                            for item in output_decoded[propType][propName]:
                                component_name = item['name']
                                component_score = item['score']
                                component_value = item['value']
                                dataDict_out[f'{propType}_component_{component_name}'] = f'{component_score} ({component_value})'
                        else:
                            dataDict_out[f'{propType}_{propName}'] = output_decoded[propType][propName]    #[propName]
    
    ## ----------------------------------------------------------------
    ## add cns-mpo calculation separately when using V22
    if 'cns-mpo' in prop_list and version == 'V22':
        URL_web_mpo = URL_web + '/cns-mpo'
        header1_mpo = 'accept: application/json'
        dataJson_mpo = '{"inputFormat": "smiles", "outputFormat": "mrv", "structure": "%s"}' % (smi)
        commandLine_mpo = ['curl', '-X', 'POST', URL_web_mpo, '-H', header1, '-H', header2, '-d', str(dataJson_mpo)]
        try:
            output_mpo, error_mpo = subprocess.Popen(commandLine_mpo, stdout=subprocess.PIPE).communicate()
            output_mpo_decoded = ast.literal_eval(output_mpo.decode())
            dataDict_out['cns-mpo_score'] = output_mpo_decoded["score"]
        except Exception as e:
            print(f"Can not calculate cns-mpo score")
            dataDict_out['cns-mpo_score'] = np.nan
    ## ----------------------------------------------------------------
    return dataDict_out

## ===========================================================================
## ===========================================================================
## ===========================================================================
def Step_2_calculate_ChemAxon_Properties(dataTable, colName_smi, prop_list=None, version='V22', col_rename_dict=None):
    ## run calculation via API
    dataDict_prop = {}
    for idx in dataTable.index:
        smi_query = dataTable[colName_smi][idx]
        dict_result = calc_mol_ChemAxon_property(smi_query, prop_list=prop_list, version=version, detailedInfo=False)
        dataDict_prop[idx] = {}
        dataDict_prop[idx][colName_smi] = smi_query
        dataDict_prop[idx].update(dict_result)
    dataTable_prop = pd.DataFrame.from_dict(dataDict_prop).T
    dataTable_prop = dataTable_prop.rename(col_rename_dict)
    ## merge
    dataTable_merge = pd.merge(left=dataTable, right=dataTable_prop, on=colName_smi, how='left')
    return dataTable_merge

################################################################################################
################################## Kymera models ###############################################
################################################################################################
def correct_bpKa1(row, colName_pKa, ML_model=None):
    try:
        w, b = ML_model.coef_[0][0], ML_model.intercept_[0]
    except:
        w, b = 0.66, 1.47

    try:
        bpKa1_cx = row[colName_pKa]
        bpKa1_cx_new = bpKa1_cx * w + b
    except:
        bpKa1_cx_new = np.nan
    return bpKa1_cx_new
    
def correct_logD(row, colName_logD, colName_mw, ML_model_D=None, ML_model_S=None):
    try:
        w_D, b_D = ML_model_D.coef_[0][0], ML_model_D.intercept_[0]
        w_S, b_S = ML_model_S.coef_[0][0], ML_model_S.intercept_[0]
    except:
        w_D, b_D = 0.4506112569023748, 2.962839874950756
        w_S, b_S = 0.65, 1.26

    try:
        logD_cx, MW = row[colName_logD], row[colName_mw]
        if MW >= 600:
            logD_kt = logD_cx * w_D + b_D
        else:
            logD_kt = logD_cx * w_S + b_S
    except:
        logD_kt = np.nan
    return logD_kt

def Step_3_calculate_KT_Properties(dataTable, colName_cx_bpKa1='ChemAxon pKa (bpKa1)', colName_cx_logD='ChemAxon logD (pH=7.4)', colName_mw='Molecular Weight'):
    # with open("model_all.pkl", "rb") as imfh:
    #     ML_model = pickle.load(imfh)

    dataTable['Corr_ChemAxon_bpKa1'] = dataTable.apply(lambda row: correct_bpKa1(row, colName_cx_bpKa1, ML_model=None), axis=1)
    dataTable['Kymera ClogD (v1)'] = dataTable.apply(lambda row: correct_logD(row, colName_cx_logD, colName_mw, ML_model_D=None, ML_model_S=None), axis=1)
    return dataTable

################################################################################################
################################## project split ###############################################
################################################################################################
## ---------------------- clean project ------------------------------------
def cleanUp_projs(proj, sep=';'):
    list_proj = proj.split(sep)
    proj_clean = 'TBD'
    projs_key = ['IRAK4', 'TYK2', 'STAT-6', 'IRF5', 'CDK2', 'MDM2', 'SMARCA2']
    projs_early = ['FcRn', 'FEM1B', 'ACVR2', 'STAT-3', 'IGG', 'ASGPR', 'CBL-C', 'ZER1', 'BOB1', 'RAF', 'STAT-4', 'NIK']
    projs_MGD = ['CDK2 MGD', 'CRBN', 'CRBN MGD Library', 'DCAF1_MGD_Library ', 'E3 MGD Library', 'VHL', 'TRAF4']
    projs_screen = ['Reference compounds', 'Screening Library', 'TRIM58', 'Bcl-xL']
    projs_sele = projs_key + projs_early + projs_MGD + projs_screen
    for p in projs_sele:
        if p in list_proj:
            proj_clean = p
            break

    if proj_clean == 'TBD':
        proj_clean = sorted(list_proj, reverse=False)[0]
    return proj_clean 

def _define_output_filename(dataTable_proj, proj):
    protocolName = 'InSilico_PhysChem_data'
    dateToday = datetime.datetime.today().strftime('%Y%b%d')
    num_rows = dataTable_proj.shape[0]    
    fileName_out = f"{protocolName}_{proj}_{num_rows}_{dateToday}.csv"
    return fileName_out

def Step_4_save_to_cvs(dataTable, colName_prj):
    colName_prj_new = "Projects_main"
    dataTable[colName_prj_new] = dataTable[colName_prj].apply(lambda proj: cleanUp_projs(proj, sep=';'))

    ProjectList = dataTable[colName_prj_new].unique()
    # print(ProjectList)

    proj_subfolder = folderChecker(f"./dataTableByProject")

    for proj in ProjectList:
        dataTable_proj = dataTable[dataTable[colName_prj_new]==proj]

        ## define file name & path
        fileName_out = _define_output_filename(dataTable_proj, proj)
        filePath_out = f"{proj_subfolder}/{fileName_out}"

        ## save to file
        if os.path.exists(filePath_out):
            print(f'\tWarning! File {filePath_out} exists')
        else:
            dataTable_proj.to_csv(filePath_out, index=False)
            print(f'\t{len(dataTable_proj)} mols saved into: {filePath_out}')

################################################################################################
################################################################################################
################################################################################################
def main():
    ## ------------------------------------------------------------------
    print(f"==> Step 0: load the parameters ... ")
    args = Step_0_load_args()
    tmp_folder = folderChecker(f"./tmp")
    
    ## ------------------------------------------------------------------
    print(f"==> Step 1: download/load data from D360 ... ")
    # '4745'
    dataTable = Step_1_load_data_table(args)
    dataTable.to_csv('./tmp/original_data_cleaned.csv', index=False)

    ## ------------------------------------------------------------------
    print(f"==> Step 2: calculate ChemAxon properties ... ")
    ## define the api param dictionary
    prop_list = ["pka", "logd", "charge", "logp", "polar-surface-area", "hbda", "solubility", "cns-mpo", "topology-analyser"]
    
    col_rename_dict = None
    col_rename_dict2 = {
        'fsp3': 'Fsp3', 
        'rotatableBondCount': 'Rotatable bonds', 
        'aromaticRingCount': '',
        'chiralCenterCount': '',
                'Polar surface area': 'ChemAxon PSA',
                'logP': 'ChemAxon logP',
                'logD[pH=7.4]': 'ChemAxon logD (pH=7.4)',
                'apKa1': 'ChemAxon pKa (apKa1)', 
                'apKa2': 'ChemAxon pKa (apKa2)', 
                'bpKa1': 'ChemAxon pKa (bpKa1)', 
                'bpKa2': 'ChemAxon pKa (bpKa2)',
                'acidicValuesByAtom': 'ChemAxon pKa (acidicValuesByAtom)',
                'basicValuesByAtom': 'ChemAxon pKa (basicValuesByAtom)'
                }    

    ## run calculation
    version = 'V22'
    colName_smi = args.colName_smi
    prop_list = ["pka", "logd", "charge", "logp", "polar-surface-area", "hbda", "solubility", "cns-mpo", "topology-analyser"]
    col_rename_dict = {'charge_formalCharge': "ChemAxon formalCharge", 'logp_logP': "ChemAxon logP",
                       'pka_apKa1': 'ChemAxon pKa (apKa1)', 'pka_apKa2': 'ChemAxon pKa (apKa2)',
                       'pka_bpKa1': 'ChemAxon pKa (bpKa1)', 'pka_bpKa2': 'ChemAxon pKa (bpKa2)',
                       'logd[pH=1.5]': 'ChemAxon logD (pH=1.5)', 'logd[pH=5.0]': 'ChemAxon logD (pH=5.0)', 
                       'logd[pH=6.5]': 'ChemAxon logD (pH=6.5)', 'logd[pH=7.4]': 'ChemAxon logD (pH=7.4)', 
                       'hbda_donorAtomCount': "ChemAxon HBD_Atom", 'hbda_acceptorAtomCount': "ChemAxon HBA_Atom", 
                       'hbda_donorSiteCount': "ChemAxon HBD_Site", 'hbda_acceptorSiteCount': "ChemAxon HBA_Site", 
                       'polar-surface-area_polarSurfaceArea': "ChemAxon PSA", 
                       'solubility(logS)': "ChemAxon Solubility(logS)", 
                       'topology-analyser_fsp3': "ChemAxon fSP3", 
                       'topology-analyser_rotatableBondCount': "ChemAxon Rotatable_Bond_Count", 
                       'topology-analyser_chainBondCount': "ChemAxon Chain_Bond_Count", 
                       'topology-analyser_chiralCenterCount': "ChemAxon Chiral_Center_Count",
                       'topology-analyser_aromaticRingCount': "ChemAxon Aromatic_Ring_Count",
                       'topology-analyser_heteroRingCount': "ChemAxon Hetero_Ring_Count",
                       'cns-mpo_score': "ChemAxon CNS_MPO_Score"}

    dataTable = Step_2_calculate_ChemAxon_Properties(dataTable, colName_smi, prop_list=prop_list, version=version, col_rename_dict=col_rename_dict)
        
    dataTable.to_csv(f'{tmp_folder}/cx_calculations.csv', index=False)
    
    ## ------------------------------------------------------------------
    print(f"==> Step 3: calculate the Kymera models ... ")
    dataTable = Step_3_calculate_KT_Properties(dataTable, colName_cx_bpKa1='ChemAxon pKa (bpKa1)', colName_cx_logD='ChemAxon logD (pH=7.4)')
    dataTable.to_csv(f'{tmp_folder}/All_calculations.csv', index=False)
    
    ## ------------------------------------------------------------------
    print(f"==> Step 4: loop the projects and extract/save data table into separated files ... ")
    Step_4_save_to_cvs(dataTable, colName_prj=args.colName_prj)

if __name__ == '__main__':
    main()
