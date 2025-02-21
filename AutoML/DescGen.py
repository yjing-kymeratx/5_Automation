'''
    -i      '../../1_DataPrep/results/data_input_clean.csv'
    -d      ','
    --colId     'Compound Name'
    --colSmi        'Structure'
    --desc_rdkit        True 
    --desc_fps      True 
    --desc_cx       True
'''

####################################################################
################## RDKit property calculator #######################
####################################################################
class desc_calculator_rdkit(object):
    def __init__(self, physChem=True, subStr=True, clean=False):
        self._desc_physChem = physChem
        self._desc_subStr = subStr
        self._desc_clean = clean
        self._desc_list = self.__define_desc_list()

        from rdkit.ML.Descriptors import MoleculeDescriptors
        self._calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self._desc_list)
        # print(f"\tInitiate a RDKit desc calcualtor for {len(self._desc_list)} desc.")

    def __define_desc_list(self):
        from rdkit.Chem import Descriptors

        ## error checking
        assert self._desc_physChem or self._desc_subStr, f"\Error! One of <physChem> or <subStr> should be True."

        # all descriptors (210)
        all_list = [n[0] for n in Descriptors._descList]
        
        ## define descriptor list
        if self._desc_physChem and self._desc_subStr:
            # using all descriptors (210)
            desc_list = all_list         
        elif self._desc_physChem and not self._desc_subStr:
            # only using 125 physicochemical properties
            desc_list = [i for i in all_list if not i.startswith('fr_')]   
        
        elif not self._desc_physChem and self._desc_subStr:
            # only use 85 substructure features <Fraction of a substructure (e.g., 'fr_Al_COO')>
            desc_list = [i for i in all_list if i.startswith('fr_')]

        if self._desc_clean:
            list_rm_prefix = ['BCUT2D_', 'Chi', 'EState_', 'VSA_', 'SlogP_', 'SMR_', 'PEOE_']
            for rm_prefix in list_rm_prefix:
                desc_list = [i for i in desc_list if not i.startswith(rm_prefix)]
        return desc_list

    def calculate(self, smi):
        self.dataDict_results = {}
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
        except Exception as e:
            print(f'\tThis SMILES cannot be transfer into mol using RDKit: {smi}; Error msg: {e}')
        else:
            try:
                result = self._calculator.CalcDescriptors(mol)
            except Exception as e:
                print(f'\tThis mol cannot be calculated property using RDKit; Error msg: {e}')
            else:
                assert len(self._desc_list) == len(result), f"\tError! Num_calc_desc does not match desc_list"
                
                for i in range(len(self._desc_list)):
                    desc_name = f"rd_{self._desc_list[i]}"
                    self.dataDict_results[desc_name] = result[i]
        return None


####################################################################
############### Mol Fingerprints calculator ########################
####################################################################
class desc_calculator_morganFPs(object):
    def __init__(self, radius=3, nBits=1024):
        self._radius = int(radius)
        self._nBits = int(nBits)

    def calculate(self, smi):        
        self.dataDict_results = {}
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
        except Exception as e:
            print(f'\tThis SMILES cannot be transfer into mol using RDKit: {smi}; Error msg: {e}')
        else:
            try:
                from rdkit.Chem import AllChem, rdMolDescriptors
                # result = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self._nBits)
                result = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self._nBits)
            except Exception as e:
                print(f'\tThis mol cannot be calculated into FPs using RDKit; Error msg: {e}')
            else:
                # dataDict_results['Fps'] = Fps
                for i in range(len(result)):
                    desc_name = f"fp_{i}"
                    self.dataDict_results[desc_name] = result[i]
        return None

   
####################################################################
############### ChemAxon property calculator########################
####################################################################
class desc_calculator_chemaxon(object):
    def __init__(self, version='V22', desc_list=None):
        self._define_cxAPI(version)
        self._define_desc_list(desc_list)

    ## ==================== define the calculator ====================
    def _define_cxAPI(self, version):
        self._version = version
        if version == 'V23':    ## v23.16
            ip = '172.31.19.252'
        elif version == 'V22':    ## v 22.50
            ip = '172.31.25.202'
        else:    ## 22.50
            ip = '172.31.25.202'

        URL_api = f'http://{ip}:8064/rest-v1/calculator/calculate' 
        header1 = 'accept: */*'    # header1 = 'accept: application/json'
        header2 = 'Content-Type: application/json'
        self._api = ['curl', '-X', 'POST', URL_api, '-H', header1, '-H', header2]
        return None

    def _define_desc_list(self, desc_list):
        api_param_dict = self._load_api_param_dict()

        if desc_list == 'all' or desc_list is None:
            self._desc_list = list(api_param_dict.keys())
        elif desc_list == 'basic':
            self._desc_list = ["elemental-analysis", "polar-surface-area", "hbda", "hlb", "solubility"]
        elif desc_list == 'protonation':
            self._desc_list = ["logp", "logd", "charge", "pka"]
        elif desc_list == 'topology':
            self._desc_list = ["topology-analyser"]
        else:
            self._desc_list = []
            for desc in desc_list:
                if desc in api_param_dict:
                    self._desc_list.append(desc)
                else:
                    print(f"\t\tWarning, this prop <{desc}> is not in the <calculations dict>")

        ## define api_param
        self._api_param = []
        for d in self._desc_list:
            desc_param = api_param_dict[d]
            self._api_param.append(f'"{d}": {desc_param}')
        return None
    
    ## ==================== run the API call ====================
    def _run_cxAPI(self, mol):
        ## ---------------- prepare dataJson & cmd ----------------
        ## prepare smiles
        from rdkit import Chem
        smi = Chem.MolToSmiles(mol, canonical=True)

        ## prepare dataJson
        api_param = ', '.join(self._api_param)
        dataJson = '{"calculations": {%s}, "inputFormat": "smiles", "structure": "%s"}' % (api_param, smi)        

        ## ---------------- run command ----------------
        import subprocess
        self._cmd = self._api + ['-d', str(dataJson)]
        process = subprocess.Popen(self._cmd, stdout=subprocess.PIPE)
        output, error = process.communicate()
        return (output, error)

    ## ==================== decode the calculation output ====================
    def _parse_result(self, result, detailedInfo=False):
        output, error = result[0], result[1]
        dataDict_out = {}
        try:
            import ast
            output_decoded = ast.literal_eval(output.decode())
        except Exception as e:
            print(f'\tCannot decode the output. Error msg: {e}')
        else:
            ## loop all the results in the output and extract them one by one
            for propType in output_decoded:
                ## eliminate un-wanted properties
                if propType not in ["isoelectric-point", "pka-distribution", "major-microspecies"]:
                    ## detect if there is errors for the calculation
                    if "error" in output_decoded[propType]:
                        dataDict_out[propType] = output_decoded[propType]["error"]["message"]
                    
                    elif propType == 'logd':
                        for item in output_decoded[propType]['logDByPh']:
                            pH = item['pH']
                            dataDict_out[f'{propType}[pH={pH}]'] = item['value']
                    
                    elif propType == 'solubility':
                        if 'unit' in output_decoded[propType]:
                            propType_specific = propType + '(' + output_decoded[propType]['unit'] + ')'
                        else:
                            propType_specific = propType
                        for item in output_decoded[propType]:
                            if item == 'intrinsicSolubility':
                                 dataDict_out[item] = output_decoded[propType][item]
                            elif item == 'phDependentSolubilities':
                                for sol in output_decoded[propType][item]:
                                    this_ph, this_value = sol['pH'], sol['value']
                                    dataDict_out[f'{propType_specific}[pH={this_ph}]'] = this_value
                
                    elif propType == 'pka':
                        ## all pKa value list and atom info
                        for pKa_type in ['acidic', 'basic']:
                            pKa_spec = sorted(output_decoded[propType][f'{pKa_type}ValuesByAtom'], key=lambda x: x['value'], reverse=False)
                            for i_pka in range(min([len(pKa_spec), 2])):
                                pKa_colName = [f'{pKa_type[0]}pKa1', f'{pKa_type[0]}pKa2'][i_pka]
                                dataDict_out[f'pka_{pKa_colName}'] = pKa_spec[i_pka]['value']
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
                                dataDict_out[f'{propType}_{propName}'] = output_decoded[propType][propName]            
        return dataDict_out

    ## run calculation
    def calculate(self, smi):
        self.dataDict_results = {}
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
        except Exception as e:
            print(f'\tThis SMILES cannot be transfer into mol using RDKit: {smi}; Error msg: {e}')
        else:
            try:
                result = self._run_cxAPI(mol)
                dataDict_out =  self._parse_result(result, detailedInfo=False)
            except Exception as e:
                print(f'\tThis mol cannot be calculated property using ChemAxon; Error msg: {e}')
            else:
                for desc in dataDict_out:
                    if desc not in ["elemental-analysis_formula", "polar-surface-area_unit"]:
                        desc_name = f"cx_{desc}"
                        self.dataDict_results[desc_name] = dataDict_out[desc]
        return None

    ## define the API calculation parameters
    def _load_api_param_dict(self):
        api_param_dict = {}

        ## --------------- basic ---------------
        api_param_dict["elemental-analysis"] = '{"countAtoms": [1, 6, 8], "countIsotopes": [{"atomNumber": 6, "isotopeNumber": 12}], "operations": "mass, formula", "symbolID": true}'
        # api_param_dict["partial-elemental-analysis"] = '{"indexes":[0]}'
        api_param_dict["polar-surface-area"] = '{"excludePhosphorus": true, "excludeSulfur": true, "pH": null}'
        
        if self._version == 'V23':
            api_param_dict["hbda"] = '{"excludeHalogens": true, "excludeSulfur": true, "outputFormat": "mrv", "outputStructureIncluded": false, "pH": 7.4}'
        elif self._version == 'V22':
            api_param_dict["hbda"] = '{"excludeHalogens": true, "excludeSulfur": true, "resultMoleculeFormat": "mrv", "resultMoleculeIncluded": false, "pH": 7.4}'
        else:
            api_param_dict["hbda"] = '{"excludeHalogens": true, "excludeSulfur": true, "resultMoleculeFormat": "mrv", "resultMoleculeIncluded": false, "pH": 7.4}'

        ## --------------- protonation ---------------
        api_param_dict["logp"] = '{"atomIncrements": true, "method": "CHEMAXON"}'
        api_param_dict["logd"] = '{"phList": [7.4]}'    # 'logd': '{"phList": [1.5, 5, 6.5, 7.4]}'
        api_param_dict["charge"] = '{"ph": 7.4}'
        # api_param_dict["pka-distribution"] = '{"considerTautomerization": true, "pKaLowerLimit": -20, "pKaUpperLimit": 10, "phSequence": {"pHLower": 1.5, "pHStep": 0.1, "pHUpper": 7.4}, "resultMoleculeFormat": "MRV", "temperature": 298}',

        if self._version == 'V23':
            api_param_dict["pka"] = '{"micro": false, "outputFormat": "mrv", "outputStructureIncluded": false, "pKaLowerLimit": -20, "pKaUpperLimit": 10, "prefix": "DYNAMIC", "temperature": 298, "types": "pKa, acidic, basic"}'
        elif self._version == 'V22':
            api_param_dict["pka"] = '{"micro": false, "pKaLowerLimit": -10, "pKaUpperLimit": 20, "prefix": "STATIC", "temperature": 298, "types": "pKa, acidic, basic"}'
        else:
            api_param_dict["pka"] = '{"micro": false, "pKaLowerLimit": -10, "pKaUpperLimit": 20, "prefix": "STATIC", "temperature": 298, "types": "pKa, acidic, basic"}'
            # api_param_dict["pka"] = '{"micro": false, "pKaLowerLimit": -20, "pKaUpperLimit": 10, "prefix": "DYNAMIC", "temperature": 298, "types": "pKa, acidic, basic"}'
        
        ## --------------- topology (ring system) ---------------
        # myOperationTopology = "aromaticRingCount, aromaticRings"
        if self._version == 'V23':
            myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize, ringSystemCount'
            api_param_dict["topology-analyser"] = '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "ringSize": 0, "ringSystemSize": 0, "operations": "myOperationText", "outputFormat": "mrv"}'
        elif self._version == 'V22':
            myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize'
            api_param_dict["topology-analyser"] = '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "ringSize": 0, "ringSystemSize": 0, "operations": "myOperationText"}'
        else:
            myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize'
            api_param_dict["topology-analyser"] = '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "ringSize": 0, "ringSystemSize": 0, "operations": "myOperationText"}'
        api_param_dict["topology-analyser"] = api_param_dict['topology-analyser'].replace('myOperationText', myOperationTopology)
       
        ## --------------- prediction ---------------
        api_param_dict["hlb"] = '{}'
        api_param_dict["solubility"] = '{"phSequence": {"pHLower": 7.4,"pHStep": 0.1,"pHUpper": 7.4}, "unit": "LOGS"}'
        # api_param_dict["bbb"] = '{}'
        # api_param_dict["cns-mpo"] = '{}'
        # api_param_dict["herg-activity"] = '{"outputFormat": "mrv"}'
        # api_param_dict["herg-class"] = '{"outputFormat": "mrv"}'


        ## --------------- 3D Conformation ---------------
        # api_param_dict["conformer"] = '{"conformerCount": 5, "diversity": 0.1, "outputFormat": "mrv", "timeLimit": 900}'
        
        ## --------------- others ---------------
        api_param_dict["isoelectric-point"] = '{"pHStep": 0.5}'
        api_param_dict["major-microspecies"] = '{"pH": 7.4, "resultMoleculeFormat": "MRV"}'

        ## --------------- unlicensed ---------------
        # api_param_dict["stereoisomer"] = '{"maxStereoisomerCount": 1000, "outputIn3d": false, "protectDoubleBondStereo": false, "protectTetrahedralStereo": false, "resultMoleculeFormat": "MRV", "type": "TETRAHEDRAL", "verify3d": false}'
        # api_param_dict["tautomerization-canonical"] = '{"normalTautomerGenerationMode": true, "resultMoleculeFormat": "MRV"}'
        # api_param_dict["tautomerization-dominant"] = '{"resultMoleculeFormat": "MRV"}'

        return api_param_dict


####################################################################
######################## Normalization #############################
####################################################################
def _calc_z_score(value_list):
    import numpy as np
    value_array = np.array(value_list)
    v_mean = np.mean(value_array)
    v_sd = np.std(value_array)
    return v_mean, v_sd

def descriptor_norm(dataTable, colName_mid):
    dataTable_desc = dataTable.drop(columns=[colName_mid])
    for col in dataTable_desc.columns:
        try:
            desc_values = dataTable_desc[col].to_numpy()
            v_mean, v_sd = _calc_z_score(desc_values)
            dataTable_desc[col] = (dataTable_desc[col] - v_mean)/ v_sd
        except Exception as e:
            print(f"Warning! This desc <{col}> cannot be normalized using Z-score! Error msg: {e}")
    dataTable_desc[colName_mid] = dataTable[colName_mid]
    return dataTable_desc



####################################################################
########################## Tools ###################################
####################################################################
## get the args
def Args_Prepation(parser_desc):
    import argparse
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument('-i', '--input', action="store", default=None, help='The input csv file')
    parser.add_argument('-d', '--delimiter', action="store", default=',', help='The delimiter of input csv file for separate columns')
    # parser.add_argument('--detectEncoding', action="store_true", help='detect the encoding type of the csv file')
    parser.add_argument('--colId', action="store", default='Compound Name', help='The column name of the compound identifier')
    parser.add_argument('--colSmi', action="store", default='Structure', help='The column name of the compound smiles')

    parser.add_argument('--desc_fps', action="store", default="True", help='calculate the molecular fingerprints')
    parser.add_argument('--desc_rdkit', action="store", default="True", help='calculate the molecular property using RDKit')
    parser.add_argument('--desc_cx', action="store", default="True", help='calculate the molecular property using ChemAxon')

    parser.add_argument('--norm', action="store", default="True", help='normalize the descriptors (z-score)')
    parser.add_argument('-o', '--output', action="store", default="./results", help='the output folder')

    args = parser.parse_args()
    return args

## calculate the descs
def calc_desc_for_table(dataTable, colName_mid, colName_smi, desc_calculator):
    dataDict_desc = {}

    for idx in dataTable.index:
        if idx % 500 == 0:
            print(f"\t\tIndex {idx}")
        mid, smi = dataTable[colName_mid][idx], dataTable[colName_smi][idx]
        ## initiate the dict
        if mid not in dataDict_desc:
            dataDict_desc[mid] = {}
            dataDict_desc[mid][colName_mid] = mid
        ## run the calculation
        desc_calculator.calculate(smi)
        if len(desc_calculator.dataDict_results) > 0:
            dataDict_desc[mid].update(desc_calculator.dataDict_results)

    return dataDict_desc

## Suppress RDKit warnings
def mute_rdkit():
    from rdkit import RDLogger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

####################################################################
######################### main function ############################
####################################################################
def main():
    import pandas as pd

    args = Args_Prepation(parser_desc='Preparing the input files and the descriptors')
    fileNameIn = args.input    # '../../1_DataPrep/results/data_input_clean.csv'
    sep = args.delimiter    # ',' 
    # detect_encoding = True if args.detectEncoding else False
    colName_mid = args.colId    # 'Compound Name'
    colName_smi = args.colSmi    # 'Structure'
    desc_fps = True if args.desc_fps=="True" else False
    desc_rdkit = True if args.desc_rdkit=="True" else False
    desc_cx = True if args.desc_cx=="True" else False
    do_norm = True if args.norm=='True' else False
    folderPathOut = args.output    ## './results'

    ## descriptor calculation params
    rd_physChem, rd_subStr, rd_clean = True, True, True
    fp_radius, fp_nBits = 3, 2048
    cx_version, cx_desc = 'V22', 'all'

    ## ------------ load data ------------
    dataTable_raw = pd.read_csv(fileNameIn, sep=sep)
    print(f"\t{dataTable_raw.shape}")
    assert colName_mid in dataTable_raw.columns, f"\tColumn name for mol ID <{colName_mid}> is not in the table."
    assert colName_smi in dataTable_raw.columns, f"\tColumn name for mol smiles <{colName_smi}> is not in the table."

    result_dict = {}
    print(f"\tCalculating descriptors (RDKit: {desc_rdkit}; FPs {desc_fps}; ChemAxon {desc_cx}) ... ")
    ## ------------ calculate rdkit properties ------------
    if desc_rdkit:
        mute_rdkit()
        print(f"\tNow calculating the rdkit descriptors")
        calculator_rd = desc_calculator_rdkit(physChem=rd_physChem, subStr=rd_subStr, clean=rd_clean)
        result_dict['rdkit'] = calc_desc_for_table(dataTable_raw, colName_mid, colName_smi, calculator_rd)

    ## ------------ calculate mol fingerprints ------------
    if desc_fps:
        mute_rdkit()
        print(f"\tNow calculating the molecular fingerprints descriptors")
        calculator_fp = desc_calculator_morganFPs(radius=fp_radius, nBits=fp_nBits)
        result_dict['fingerprints'] = calc_desc_for_table(dataTable_raw, colName_mid, colName_smi, calculator_fp)

    ## ------------ calculate chemAxon properties ------------
    if desc_cx:
        print(f"\tNow calculating the chemaxon descriptors")
        calculator_cx = desc_calculator_chemaxon(version=cx_version, desc_list=cx_desc)
        result_dict['chemaxon'] = calc_desc_for_table(dataTable_raw, colName_mid, colName_smi, calculator_cx)

    ## ------------ save output ------------
    for app in result_dict:
        data_table = pd.DataFrame.from_dict(result_dict[app]).T

        ## normalization
        if do_norm and app not in ['fingerprints']:
            data_table_norm = descriptor_norm(data_table, colName_mid=colName_mid)
        else:
            data_table_norm = data_table

        ## save to csv
        import os
        os.makedirs(folderPathOut, exist_ok=True)

        out_csv = f"./results/descriptors_{app}.csv"
        data_table.to_csv(out_csv, index=False)
        print(f"\tThe <{app}> descriptor data {data_table.shape} has been saved to <{out_csv}>.")

        out_csv_norm = f"./results/descriptors_{app}_norm.csv"
        data_table_norm.to_csv(out_csv_norm, index=False)
        print(f"\tThe normalized <{app}> descriptor data {data_table_norm.shape} has been saved to <{out_csv_norm}>.")

        
    # return result_dict

if __name__ == '__main__':
    main()