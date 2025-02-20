import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

########################################################################################
####################### Custom functions/tools to process data #########################
########################################################################################
## ---------------- tool to transfer between smiles and RDKit mol objects ----------------

def tool_mol_2_smi(mol):
    try:
        smi = Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(e)
        smi = np.nan
    return smi

## ---------------- tool to generate <dataJson> for API call ----------------
def _prep_dataJson(smi_query, api_param_dict, proplist=['pka']):
    ## based on the query calculation, prepare the calculators (string)
    dataList_calculators = []
    for prop in proplist:
        if prop in api_param_dict:
            prop_param = api_param_dict[prop]
            dataList_calculators.append(f'"{prop}": {prop_param}')
        else:
            print(f"Warning, this prop <{prop}> is not in the <calculations dict>")
    api_calculator = ', '.join(dataList_calculators)
    
    ## prepare the dataJson string for API calls
    dataJson = '{"calculations": {%s}, "inputFormat": "smiles", "structure": "%s"}' % (api_calculator, smi_query)
    return dataJson

def Step_1_build_cmd(smi_query, api_param_dict, proplist=None, version='V22'):
    ## ----------------------------------------------------------------
    if version == 'V23':    ## v23.16
        URL_api = 'http://172.31.19.252:8064/rest-v1/calculator/calculate' 
    elif version == 'V22':    ## v 22.50
        URL_api = 'http://172.31.25.202:8064/rest-v1/calculator/calculate'
    else:    ## v23.16
        URL_api = 'http://172.31.19.252:8064/rest-v1/calculator/calculate' 

    header1 = 'accept: */*'    # header1 = 'accept: application/json'
    header2 = 'Content-Type: application/json'

    ## ----------------- data JSON -----------------
    if proplist is None:
        proplist = list(api_param_dict.keys())
    dataJson = _prep_dataJson(smi_query, api_param_dict, proplist)
    
    ## ----------------- Define the command you want to execute -----------------
    commandLine = ['curl', '-X', 'POST', URL_api, '-H', header1, '-H', header2, '-d', str(dataJson)]
    return commandLine

## basic run cmd
def Step_2_run_cmd(commandLine):
    ## count time
    # beginTime = time.time()

    # Use subprocess to execute the command
    import subprocess
    process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # costTime = time.time()-beginTime
    # print(f"\tThis calculation costs time = %ds ................" % (costTime))
    return (output, error)

## basic run cmd
def Step_3_analyze_results(output):
    result_dict = {}   
    try:
        import ast
        output_dict = ast.literal_eval(output.decode())
    except:
        pass
    else:
        # print(f"\t>>>>Calculated properties: {output_dict.keys()}")
        for propName in output_dict:
            prop_dict = output_dict[propName]
            ## ---------------- pKa ----------------
            if propName == 'pka':
                atoms_list = []
                ## acidic pKa
                for i in range(2):
                    try:
                        apKa_list = sorted(prop_dict['acidicValuesByAtom'], key=lambda x: x['value'], reverse=False)
                        result_dict[f"apKa{i+1}"] = apKa_list[i]['value']
                        atoms_list.append(str(apKa_list[i]['atomIndex']))
                    except:
                        result_dict[f"apKa{i+1}"] = ''    #np.nan
                        atoms_list.append('')
                ## basic pKa
                for i in range(2):
                    try:
                        bpKa_list = sorted(prop_dict['basicValuesByAtom'], key=lambda x: x['value'], reverse=True)
                        result_dict[f"bpKa{i+1}"] = bpKa_list[i]['value']
                        atoms_list.append(str(bpKa_list[i]['atomIndex']))
                    except:
                        result_dict[f"bpKa{i+1}"] = ''    #np.nan
                        atoms_list.append('')
                result_dict['pKa_atoms'] = ','.join(atoms_list)

                for pka_by_atom in ['acidicValuesByAtom', 'basicValuesByAtom', 'pkaValuesByAtom']:
                    try:
                        result_dict[pka_by_atom] = prop_dict[pka_by_atom]
                    except:
                        result_dict[pka_by_atom] = ''
            elif propName == 'charge':
                try:
                    result_dict[f"Formal charge"] = prop_dict["formalCharge"]
                except Exception as e:
                    pass
            ## ---------------- other ----------------
            else:
                pass
    return result_dict

## ===========================================================================
def calc_ChemAxon_property(smi_query, api_param_dict, proplist, version='V22'):
    ## calculate properties
    cmd_cxAPI = Step_1_build_cmd(smi_query, api_param_dict, proplist, version)
    print(cmd_cxAPI)
    (output, error) = Step_2_run_cmd(cmd_cxAPI)
    result_dict = Step_3_analyze_results(output)
    result_dict['output'] = output
    result_dict['error'] = error
    return result_dict


## ===========================================================================
def bpka1_correction(bpka1):
    try:
        bpka1_corr = 0.66*bpka1+1.47
    except:
        bpka1_corr = None
    return bpka1_corr
## ===========================================================================
def plot_pKa(smiles, pka_values_bacic, pka_values_acidic, imgFileName='test.png'):
    # import ast
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    # Generate RDKit molecule object from SMILES
    molecule = Chem.MolFromSmiles(smiles)

    highlight_atoms, highlight_colors = [], {}

    ## basic pka
    for bpka in pka_values_bacic:
        atom_idx, atom_pka = bpka['atomIndex'], bpka['value']
        atom = molecule.GetAtomWithIdx(atom_idx)
        atom.SetProp('atomNote', str(atom_pka))

        # Specify the atoms to highlight and their colors
        highlight_atoms.append(atom_idx)
        highlight_colors[atom_idx] = (0, 0, 1)

    ## acidic pka
    for apka in pka_values_acidic:
        atom_idx, atom_pka = apka['atomIndex'], apka['value']
        atom = molecule.GetAtomWithIdx(atom_idx)
        atom.SetProp('atomNote', str(atom_pka))

        # Specify the atoms to highlight and their colors
        highlight_atoms.append(atom_idx)
        highlight_colors[atom_idx] = (1, 0, 0)

    # Create a drawer object with the specified options
    d = rdMolDraw2D.MolDraw2DCairo(600, 600)

    # Create a drawing options object and set the font size
    options = Draw.MolDrawOptions()
    options.annotationFontScale = 1.5  # Set the font size for atom labels
    d.SetDrawOptions(options)

    # Draw the molecule with highlighted atoms
    # d.DrawMolecule(molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    d.DrawMolecule(molecule)


    # Draw the molecule
    d.FinishDrawing()

    # Save the image to a file
    d.WriteDrawingText(imgFileName)
    return imgFileName
########################################################################################
################################# main function ########################################
########################################################################################
def main():
    ## ================== read in the sdf file of mol & model ==================
    import argparse
    parser = argparse.ArgumentParser(description='Calculate ChemAxon property using micro-service API, details to be added later')
    parser.add_argument('-i', '--input', action="store", default=None, help='The path of input sdf file')
    parser.add_argument('-o', '--output', action="store", default="results.csv", help='The path of output csv file')
    # parser.add_argument('--api_param', action="store", default='cx_calculation_param.json', help='The json file contains the api parameters')
    parser.add_argument('--propertylist', action="store", default='pka,logp', help='comma separated string of the properties to be calculated')
    parser.add_argument('--pkaImg', action='store_true', help='Generate the molecule image with pKa values')

    args = parser.parse_args()    ## parse the arguments
    
    ## input and output
    in_sdf_path = args.input    #sdfpath = './Test.sdf'
    out_csv_path = args.output

    ## properties want to calculate
    proplist = args.propertylist.split(',')

    ## param dict
    import sys
    # sys.path.append('/fsx/home/yjing/models')
    sys.path.append('./tool')
    import cx_API_param
    api_param_dict = cx_API_param.load_api_param_dict()
    # import json
    # with open(args.api_param, 'r') as apiparamfh:
    #     api_param_dict = json.load(apiparamfh)

    ## chemaxon jchem version
    cx_version = 'V22'

    ## ================== read in the sdf file of mol & model ==================
    ## load the data
    mols = [mol for mol in Chem.SDMolSupplier(in_sdf_path)]
    colName_molID = "Corporate ID"
    colName_Smiles = 'SMILES'

    ## calculate the ChemAxon property using API
    dataDict_out = {}
    for idx in range(len(mols)):
        mol = mols[idx]
        mol_mid = mol.GetProp(colName_molID)
        mol_smi = tool_mol_2_smi(mol)

        dataDict_out[idx] = {}
        dataDict_out[idx][colName_molID] = mol_mid
        dataDict_out[idx][colName_Smiles] = mol_smi

        dataDict_out_prop = calc_ChemAxon_property(mol_smi, api_param_dict, proplist, version=cx_version)
        dataDict_out[idx].update(dataDict_out_prop)

        if 'bpKa1' in dataDict_out[idx]:
            bpka1_corr = bpka1_correction(dataDict_out[idx]['bpKa1'])
            if bpka1_corr is not None:
                dataDict_out[idx]['bpKa1_corrected'] = bpka1_corr

        ## plot
        if args.pkaImg:
            # Create the directory if it does not exist
            import os
            image_dir = "./pKa_image"
            os.makedirs(image_dir, exist_ok=True)

            ##
            if 'pka' not in proplist:
                print(f"\tWarning! Cannot generate pKa image because <pka> is not in the --propertylist")
            else:
                try:
                    pka_values_bacic = dataDict_out[idx]['basicValuesByAtom']
                    pka_values_acidic = dataDict_out[idx]['acidicValuesByAtom']
                    print(f"\t\tThere are {len(pka_values_bacic)} bpKa values and {len(pka_values_acidic)} apKa values")
                    plot_file = plot_pKa(mol_smi, pka_values_bacic, pka_values_acidic, imgFileName=f"{image_dir}/{mol_mid}.png")
                except Exception as e:
                    print(f"Cannot generate pKa image for <{mol_mid}> because: {e}")
                else:
                    dataDict_out[idx]['pka_image'] = plot_file

    dataTable_out = pd.DataFrame.from_dict(dataDict_out).T
    
    ## output datatable
    dataTable_out.to_csv(out_csv_path, index=False)

if __name__ == '__main__':
    main()