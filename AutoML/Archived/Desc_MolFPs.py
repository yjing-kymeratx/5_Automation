from rdkit import Chem
from rdkit.Chem import AllChem

####################################################################
####################################################################
####################################################################
class morgan_Fps_calculator(object):
    ## <----- model initiation ---->
    def __init__(self, radius=3, nBits=1024):
        self._radius = int(radius)
        self._nBits = int(nBits)

    def calculation_from_mol(self, mol):
        try:
            Fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self._nBits)
        except Exception as e:
            print(f'\tThis mol cannot be calculated into FPs using RDKit; Error msg: {e}')
            dataDict_results = None
        else:
            dataDict_results = {}
            dataDict_results['Fps'] = Fps
            for i in range(len(Fps)):
                dataDict_results[f'FP_bit_{i}'] = Fps[i]
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
    
    def _cleanup_smi(self, smi):
        if "\\" in smi:
            print(f'\tThere is a "\\" in the SMILES {smi}')
            smi = smi.replace('\\', '\\\\')
            print(f'\tAdd 1 "\\" into the SMILES, now new SMILES is {smi}')
        return smi
####################################################################
####################################################################
####################################################################
## calc mol FPs
def calc_desc_fingerprints(molDict, fpType="ECFP", radius=3, nBits=2048):
    ## initiate a fps calculator
    if fpType == "ECFP":
        fpsCalculator = morgan_Fps_calculator(radius=radius, nBits=nBits)
    else:
        print(f"\tWarning! Current version only support ECFP. Now generating the default ECFP{radius*2} ({nBits}bits)")
        fpsCalculator = morgan_Fps_calculator(radius=radius, nBits=nBits)

    ## loop through the mol list and calculate the fps
    print(f'\t----------- Now start calculating Molecular Fingerprints ----------')
    for cid in molDict:
        molDict[cid]['desc_fps'] = {}
        smi = molDict[cid]['Smiles_clean'] if 'Smiles_clean' in molDict[cid] else molDict[cid]['Smiles']
        try:
            dataDict_fps = fpsCalculator.calculation_from_smi(smi)
        except Exception as e:
            print(f"\tWarning, the mol <{cid}> fails to calculate molecular fingerprints. Error: {e}")
        else:
            molDict[cid]['desc_fps'].update(dataDict_fps)
    print(f'\t----------- Molecular Fingerprints calculation done ----------')
    return molDict