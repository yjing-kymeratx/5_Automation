import ast
import subprocess
from rdkit import Chem

####################################################################
####################################################################
####################################################################
class ChemAxonAPI(object):
    ## <----- model initiation ---->
    def __init__(self, ip='172.31.19.252', port='8064', calculator='calculate'):
        
        self._api_url = f'http://{ip}:{port}/rest-v1/calculator/{calculator}'
        self._headers = ['accept: */*', 'Content-Type: application/json']

    ## <----- run api calls and prase the results ---->
    def calculation_from_smi(self, smi, detailedInfo=False, rmProps=[]):
        ## clean up smiles
        smi_new = smi
        # smi_new = self._cleanup_smi(smi)

        ## 1. perpare dataJson using <_generate_dataJson> function for the ChemAxon API calculation
        dataJson = self._generate_dataJson(smi_new)

        ## 2. Define the command you want to execute
        commandLine = ['curl', '-X', 'POST', self._api_url, '-H', self._headers[0], '-H', self._headers[1], '-d', str(dataJson)]

        ## 3. run the cmd using subprocess package to execute the command
        process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
        self.output, self.error = process.communicate()

        ## 4. clean up the results and
        dataDict_results = {}
        dataDict_out= self._parse_output(detailedInfo=detailedInfo)
        for prop in dataDict_out:
            if prop not in rmProps:
                dataDict_results['cx_'+prop] = dataDict_out[prop]
        return dataDict_results
    
    def calculation_from_mol(self, mol, detailedInfo=False, rmProps=[]):
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'\tThis mol cannot be generated SMILES using RDKit; Error msg: {e}')
            dataDict_results = {}
        else:
            dataDict_results = self.calculation_from_smi(smi, detailedInfo=detailedInfo, rmProps=rmProps)
        return dataDict_results

    ####################### tool function for api calls preparation ########################
    ## <----- api Json preparation ---->
    def _generate_dataJson(self, smi, propList=None):
        ## predefine the dataJson
        calculations = {
            # 'elemental-analysis': '{"countAtoms": [1, 6, 8],  "countIsotopes": [{"atomNumber": 6, "isotopeNumber": 12}], "operations": "mass, formula", "symbolID": true}',
            'elemental-analysis': '{"countAtoms": [1, 6, 8],  "countIsotopes": [{"atomNumber": 6, "isotopeNumber": 12}], "operations": "mass", "symbolID": true}',
            'polar-surface-area': '{"excludePhosphorus": true, "excludeSulfur": true, "outputFormat": "mrv", "outputStructureIncluded": false, "pH": 7.4}',
            'hbda': '{"excludeHalogens": true, "excludeSulfur": true, "outputFormat": "mrv", "outputStructureIncluded": false, "pH": 7.4}',
            'logd': '{"phList": [1.5, 5, 6.5, 7.4]}',
            'logp': '{"atomIncrements": true, "method": "CHEMAXON"}',
            'topology-analyser': '{"aliphaticRingSize": 0, "aromaticRingSize": 0, "aromatizationMethod": "GENERAL", "carboRingSize": 0, "fusedAliphaticRingSize": 0, "fusedAromaticRingSize": 0, "heteroAliphaticRingSize": 0, "ringSize": 0, "heteroAromaticRingSize": 0, "heteroRingSize": 0, "operations": "myOperationText", "outputFormat": "mrv", "ringSystemSize": 0}',
            'charge': '{"ph": 7.4}',
            'pka': '{"micro": false, "outputFormat": "mrv", "outputStructureIncluded": false, "prefix": "STATIC", "pKaLowerLimit": -10, "pKaUpperLimit": 20, "temperature": 298, "types": "pKa, acidic, basic"}',
            # 'cns-mpo': '{}',
            # 'hlb': '{}', 'bbb': '{}', 'cns-mpo': '{}',
            # 'pka-distribution': '{"considerTautomerization": true, "pKaLowerLimit": -20, "pKaUpperLimit": 10, "temperature": 298, "phSequence": {"pHLower": 1.5, "pHStep": 0.1, "pHUpper": 7.4}, "resultMoleculeFormat": "MRV"}',
            # 'solubility': '{"phSequence": {"pHLower": 1.5, "pHStep": 0.1, "pHUpper": 7.4}, "unit": "MM"}'
        }
        myOperationTopology = 'fsp3, chainBondCount, rotatableBondCount, aromaticAtomCount, chiralCenterCount, aromaticRingCount, heteroRingCount, fusedAliphaticRingCount, aliphaticRingCount, fusedAromaticRingCount, heteroAromaticRingCount, fusedRingCount, largestRingSystemSize, largestRingSize, ringSystemCount'
        calculations['topology-analyser'] = calculations['topology-analyser'].replace('myOperationText', myOperationTopology)

        ## based on the query calculation, prepare the calculators (string)
        dataList_calculators = []
        for prop in calculations:
            prop_param = calculations[prop]
            dataList_calculators.append(f'"{prop}": {prop_param}')
        
        ## prepare the dataJson string for API calls
        dataJson = '{"calculations": {%s}, "inputFormat": "smiles", "structure": "%s"}' % (', '.join(dataList_calculators), smi)
        return dataJson

    def _cleanup_smi(self, smi):
        if "\\" in smi:
            print(f'\tThere is a "\\" in the SMILES {smi}')
            smi = smi.replace('\\', '\\\\')
            print(f'\tAdd 1 "\\" into the SMILES, now new SMILES is {smi}')
        return smi

    def _parse_output(self, detailedInfo=False):
        dataDict_out = {}
        try:
            output_decoded = ast.literal_eval(self.output.decode())
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
                        for item in output_decoded[propType]['phDependentSolubilities']:
                            dataDict_out[f'{propType}[pH={pH}]'] = item['value']
                
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

####################################################################
####################################################################
####################################################################
## calc ChemAxon property
def calc_desc_chemaxon(molDict, ip='172.31.19.252', port='8064', calculator='calculate', rmProps=[]):
    ## initiate a ChemAxonAPI object
    cxAPI = ChemAxonAPI(ip=ip, port=port, calculator=calculator)

    ## loop through the mol list and calculate the properties
    print(f'\t----------- Now start calculating ChemAxon Property ----------')
    for cid in molDict:
        molDict[cid]['desc_cx'] = {}
        smi = molDict[cid]['Smiles_clean'] if 'Smiles_clean' in molDict[cid] else molDict[cid]['Smiles']
        try:
            descDict_cx = cxAPI.calculation_from_smi(smi, rmProps=rmProps)
        except Exception as e:
            print(f"\tWarning, the mol <{cid}> fails to calculate ChemAxon property. Error: {e}")
        else:
            molDict[cid]['desc_cx'].update(descDict_cx)
    print(f'\t----------- ChemAxon Property calculation done ----------')
    return molDict






