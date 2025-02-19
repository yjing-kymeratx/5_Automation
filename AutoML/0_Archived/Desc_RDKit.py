from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

####################################################################
####################################################################
####################################################################
class RDKit_desc_calculator(object):
    ## <----- model initiation ---->
    def __init__(self, physChem=True, subStr=True, clean=False):        
        self._desc_list = self.__define_desc_list(physChem=physChem, subStr=subStr, clean=clean)
        self._desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self._desc_list)
        # print(f"\tInitiate a RDKit desc calcualtor for {len(self._desc_list)} desc.")

    def calculation_from_mol(self, mol):
        try:
            rdkit_desc = self._desc_calc.CalcDescriptors(mol)
        except Exception as e:
            print(f'\tThis mol cannot be calculated property using RDKit; Error msg: {e}')
            dataDict_results = None
        else:
            assert len(self._desc_list) == len(rdkit_desc), f"\tError! Num_calc_desc does not match desc_list"
            dataDict_results = {}
            for i in range(len(self._desc_list)):
                dataDict_results[self._desc_list[i]] = rdkit_desc[i]
        return dataDict_results

    def calculation_from_smi(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            # mol = Chem.MolFromSmiles(self._cleanup_smi(smi))
        except Exception as e:
            print(f'\tThis SMILES cannot be transfer into mol using RDKit: {smi}; Error msg: {e}')
            dataDict_results = None
        else:
            dataDict_results = self.calculation_from_mol(mol)
        return dataDict_results    
    
    def __define_desc_list(self, physChem=True, subStr=True, clean=False):
        ## error checking
        assert physChem or subStr, f"\Error! One of <physChem> or <subStr> should be True."

        # all descriptors (210)
        all_list = [n[0] for n in Descriptors._descList]
        
        ## define descriptor list
        if physChem and subStr:
            # using all descriptors (210)
            desc_list = all_list         
        elif physChem and not subStr:
            # only using 125 physicochemical properties
            desc_list = [i for i in all_list if not i.startswith('fr_')]   
        
        elif not physChem and subStr:
            # only use 85 substructure features <Fraction of a substructure (e.g., 'fr_Al_COO')>
            desc_list = [i for i in all_list if i.startswith('fr_')]

        if clean:
            list_rm_prefix = ['BCUT2D_', 'Chi', 'EState_', 'VSA_', 'SlogP_', 'SMR_', 'PEOE_']
            for rm_prefix in list_rm_prefix:
                desc_list = [i for i in desc_list if not i.startswith(rm_prefix)]
        return desc_list

    def _cleanup_smi(self, smi):
        if "\\" in smi:
            print(f'\tThere is a "\\" in the SMILES {smi}')
            smi = smi.replace('\\', '\\\\')
            print(f'\tAdd 1 "\\" into the SMILES, now new SMILES is {smi}')
        return smi

####################################################################
####################################################################
####################################################################
## calc RDKit property
def calc_desc_rdkit(molDict, physChem=True, subStr=True, clean=False):
    ## initiate a rdkit calculator
    rdCalculator = RDKit_desc_calculator(physChem=physChem, subStr=subStr, clean=clean)

    ## loop through the mol list and calculate the rdkit  props
    print(f'\t----------- Now start calculating RDKit props ----------')
    for cid in molDict:
        molDict[cid]['desc_rdkit'] = {}
        smi = molDict[cid]['Smiles_clean'] if 'Smiles_clean' in molDict[cid] else molDict[cid]['Smiles']
        try:
            descDict_rdkit = rdCalculator.calculation_from_smi(smi)
        except Exception as e:
            print(f"\tWarning, the mol <{cid}> fails to calculate RDKit property. Error: {e}")
        else:
            molDict[cid]['desc_rdkit'].update(descDict_rdkit)
    print(f'\t----------- RDKit props calculation done ----------')
    return molDict