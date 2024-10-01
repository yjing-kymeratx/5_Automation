# !/fsx/home/yjing/apps/anaconda3/env/yjing/bin python

##############################################################################################
##################################### load packages ###########################################
##############################################################################################
import ast
import time
import argparse
import subprocess
import numpy as np
import pandas as pd

##############################################################################################
##################################### Custom Tools ###########################################
##############################################################################################
def _buildCmd(smi_from, myMMPsDB, property=None, radius=-1):
    if property is None:
        gen_type = "generate"
        commandLine = ["mmpdb", f"{gen_type}", "--smiles", f"{smi_from}", f"{myMMPsDB}", "--radius", f"{radius}"]
        if radius in [0, 1, 2, 3, 4, 5]:
            commandLine.append("--radius")
            commandLine.append(f"{radius}")
    else:
        gen_type = "transform"
        commandLine = ["mmpdb", f"{gen_type}", "--smiles", f"{smi_from}", f"{myMMPsDB}", "-r", f"{radius}"]
        ##
        proplist = property.split(',')
        for prop in proplist:
            commandLine.append("--property")
            commandLine.append(f"{property}")
        ##
        if radius in [0, 1, 2, 3, 4, 5]:
            commandLine.append("-r")
            commandLine.append(f"{radius}")

    print(f'\tCommands:', ' '.join(commandLine))
    return commandLine

## --------------------------------------------------
def _runCmd(commandLine):
    dataDict = {}
    try:
        process = subprocess.Popen(commandLine, stdout=subprocess.PIPE)
        output, error = process.communicate()
        list_output = output.decode().split('\n')
    except Exception as e:
        print(f'\tCannot decode the output. Error msg: {e}')
    else:
        for i in range(len(list_output)):
            if list_output[i] != '':
                list_line = list_output[i].split('\t')
                if i == 0:
                    list_colNames = list_line
                    num_cols = len(list_colNames)
                else:
                    dataDict[i] = {}

                    if len(list_line) != num_cols:
                        print(f"Error, This row {i} has different number of cols to the header row, {list_output[i]}")
                    else:
                        for colid in range(len(list_colNames)):
                            col = list_colNames[colid]
                            dataDict[i][col] = list_line[colid]
    return dataDict

## --------------------------------------------------
def CleanResults(smi_from, myMMPsDB, property=None, radius=-1):
    ##
    commandLine = _buildCmd(smi_from=smi_from, myMMPsDB=myMMPsDB, property=property, radius=radius)
    dataDict = _runCmd(commandLine)
    dataTable = pd.DataFrame.from_dict(dataDict).T

    ##
    if property is None:
        renameCols = {
            'start': 'mol_start', 
            'final': 'mol_gen', 
            'constant': 'fragment_constant', 
            'from_smiles': 'fragment_from', 
            'to_smiles': 'fragment_to', 
            'r': 'radius', 
            'Rule_Info': 'Rule_Info'}
        dataTable["Rule_Info"] = dataTable["pair_from_id"] + '=>' + dataTable["pair_to_id"] + ' (N_Pairs=' + dataTable["#pairs"] + ')'

    else:
        renameCols = {
            'start': 'mol_start',
            'SMILES': 'mol_gen',
            'constant': 'fragment_constant',
            f'{property}_from_smiles': 'fragment_from', 
            f'{property}_to_smiles': 'fragment_to', 
            f'{property}_radius': 'radius', 
            f'{property}_avg': f'{property}_avg',
            'Rule_Info': 'Rule_Info',}
            
        dataTable["start"] = smi_from
        dataTable["constant"] = np.nan
        dataTable["Rule_Info"] = 'Rule_env_id: ' + dataTable[f"{property}_rule_environment_id"] + ' (N_Pairs=' + dataTable["EstFa_Rat_count"] + ')'
    ##
    dataTable_gen = dataTable[renameCols.keys()].rename(columns=renameCols)
    print(f"\tGenerate {dataTable_gen.shape[0]} analoges")
    return dataTable_gen

## --------------------------------------------------

##############################################################################################
###################################### main function #########################################
##############################################################################################

def main():
    beginTime = time.time()

    ## ------------------ argument parser ------------------ ##
    parser = argparse.ArgumentParser(description='This is the script to transform an input structure using the rules in the MMPs database')
    parser.add_argument('-i', '--input', action="store", default=None, help='The input SMILES string to start from')
    parser.add_argument('-p', '--property', action="store", default=None, help='The property to use for rule')
    parser.add_argument('-r', '--radius', action="store", default=-1, help='Fingerprint environment radius [0|1|2|3|4|5]')
    parser.add_argument('-db', '--database', action="store", default=None, help='The MMPs databse used to generate analoges')
    parser.add_argument('-o', '--output', action="store", default="MMPsGen_results", help='The name of output csv file to save the generated molecules data')
    
    args = parser.parse_args()    ## parse the arguments

    ## get parameters from arguments
    smi_from = args.input
    property = args.property
    radius = args.radius
    myMMPsDB = args.database
    fileName_out =  f"MMPsGen_results.csv"

    ## ============================ run the code ============================ ##
    ## 1. Load the raw data from csv file
    dataTable_gen = CleanResults(smi_from=smi_from, myMMPsDB=myMMPsDB, property=property, radius=radius)

    dataTable_gen.to_csv(f"{fileName_out}", index=False)
    print(f"The output file is saved to {fileName_out}\n")
    print("==> Entire analysis costs time = %ds ................\n" % (time.time()-beginTime))

if __name__ == '__main__':
    main()
