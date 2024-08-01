import numpy as np
import pandas as pd
#########################################################################################

def extractPropertyDataFromD360Table(row, colName_mod, colName_num):
    assert colName_mod in row, f'Cannot find <Mod> column with name <{colName_mod}!'
    assert colName_num in row, f'Cannot find <Mod> column with name {colName_num}'

    result = np.nan
    if row.notna()[colName_mod] and row.notna()[colName_num]:
        if row[colName_mod] == '=':
            result = row[colName_num]
    return result

#########################################################################################

def calc_mean(value_list):
    value_list_clean = []
    for v in value_list:
        if v not in [None, np.nan, '', ' ']:
            try:
                v_num = float(v)
            except Exception as e:
                print(f'Error, cannot numericalize value {v}', e)
            else:
                value_list_clean.append(v_num)
    return np.mean(value_list_clean)

def calc_eIC50_hERG(comments_str):
    # e.g., comments_str = '21.38% inhibition @ 10 ?M'
    try:
        [str_inhb, str_conc] = comments_str.split('@')
        inhb = float(str_inhb.split('%')[0])
        inhb = 0.1 if inhb < 0 else (99.99 if inhb > 100 else inhb)
        conc = float(str_conc.split('M')[0][:-1])
        eIC50 = conc*(100-inhb)/inhb
    except Exception as e:
        eIC50 = None
        if comments_str not in [' ', '/']:
            print(f'Error, cannot calc hERG eIC50 from comment data. {comments_str}')
    return eIC50

def calc_EstFa(PKF_PO, Clobs_IV, Species='Rat'):
    dict_IV_ratio = {'Rat': 90, 'Mouse': 70, 'Dog': 30, 'Monkey': 44}  
    try:
        estfa = (PKF_PO/100)/(1-(Clobs_IV/dict_IV_ratio[Species]))
    except Exception as e:
        estfa = np.nan
    return estfa

#########################################################################################
def clean_up_permeability(row, colName_dict=None):
    if colName_dict is None:
        colName_prefix = 'ADME MDCK(WT) Permeability'
        colName_dict = {}
        colName_dict['Mod'] = colName_prefix + ';Mean;' + 'A to B Papp (10^-6 cm/s);(Mod)'
        colName_dict['Num'] = colName_prefix + ';Mean;' + 'A to B Papp (10^-6 cm/s);(Num)'

    permeability = extractPropertyDataFromD360Table(row, colName_dict['Mod'], colName_dict['Num'])
    output = pd.Series([permeability])
    return output

## -------------------- Efflux -------------------
def clean_up_efflux(row, colName_dict=None):
    if colName_dict is None:
        colName_prefix = 'ADME MDCK (MDR1) efflux'
        colName_dict = {}
        colName_dict['Mod'] = colName_prefix + ';Mean;' + 'Efflux Ratio;(Mod)'
        colName_dict['Num'] = colName_prefix + ';Mean;' + 'Efflux Ratio;(Num)'

    efflux = extractPropertyDataFromD360Table(row, colName_dict['Mod'], colName_dict['Num'])
    output = pd.Series([efflux])
    return output

## -------------------- hERG -------------------
def clean_up_hERG(row, colName_dict_IC50=None, eIC50=False, colName_cmt=None):

    ## expt IC50
    colName_prefix = 'ADME Tox-manual patch hERG 34C'
    if colName_dict_IC50 is None:
        colName_dict_IC50 = {}
        colName_dict_IC50['Mod'] = colName_prefix + ';GMean;' + 'm-patch hERG IC50 [uM];(Mod)'
        colName_dict_IC50['Num'] = colName_prefix + ';Mean;' + 'm-patch hERG IC50 [uM];(Num)'
    hERG_IC50 = extractPropertyDataFromD360Table(row, colName_dict_IC50['Mod'], colName_dict_IC50['Num'])
    output = pd.Series([hERG_IC50])

    ## estimated IC50 by comments column
    if eIC50:
        if colName_cmt is None:
            colName_hERG_cmnt = colName_prefix + ';Concat;' + 'Comments'

        ## calculate eIC50
        hERG_eIC50_list = []
        if colName_hERG_cmnt in row:
            if row.notna()[colName_hERG_cmnt]:
                for cmnt in row[colName_hERG_cmnt].split(';'):
                    hERG_eIC50_list.append(calc_eIC50_hERG(cmnt))    
        hERG_eIC50 = calc_mean(hERG_eIC50_list)

        ## determine mixedIC50
        if not np.isnan(hERG_IC50):
            hERG_mixedIC50, ambitiousData = hERG_IC50, 0
        elif not np.isnan(hERG_eIC50):
            hERG_mixedIC50, ambitiousData = hERG_eIC50, 1
        else:
            hERG_mixedIC50, ambitiousData = np.nan, np.nan
        output = pd.Series([hERG_IC50, hERG_eIC50, hERG_mixedIC50, ambitiousData])

    return output

## -------------------- PK -------------------
def clean_up_PK(row, Species='Rat', colName_dict_F=None, EstFa=False, colName_dict_Cl=None):
    dict_PK_param = {
        'Rat': {'Dose_PO':'10.000 (mg/kg)', 'Dose_IV':'2.000 (mg/kg)', 'ratio':90},
        'Mouse': {'Dose_PO':'10.000 (mg/kg)', 'Dose_IV':'2.000 (mg/kg)', 'ratio':70},
        'Dog': {'Dose_PO':'3.000 (mg/kg)', 'Dose_IV':'0.500 (mg/kg)', 'ratio':30}, 
        'Monkey': {'Dose_PO':'3.000 (mg/kg)', 'Dose_IV':'0.500 (mg/kg)', 'ratio':44}}
    
    assert Species in dict_PK_param, f'Error, Species <{Species}> is not in the list [{list(dict_PK_param.keys())}] '

    ## clean up the Oral F% data
    admin_F = 'Route of Administration: PO'
    dose_F = dict_PK_param[Species]['Dose_PO']
    if colName_dict_F is None:
        colName_dict_F = {'Mod':f'ADME PK;Mean;F %;Dose: {dose_F};{admin_F};Species: {Species};(Mod)',
                          'Num':f'ADME PK;Mean;F %;Dose: {dose_F};{admin_F};Species: {Species};(Num)'}
    PKF_PO = extractPropertyDataFromD360Table(row, colName_dict_F['Mod'], colName_dict_F['Num'])
    output = pd.Series([PKF_PO])

    ## calculate the EstFa using Oral F% data and IV Cl_obs data
    if EstFa:
        admin_Cl = 'Route of Administration: IV'
        dose_Cl = dict_PK_param[Species]['Dose_IV']

        if colName_dict_Cl is None:
            colName_dict_Cl = {'Mod':f'Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: {dose_Cl};{admin_Cl};Species: {Species};(Mod)', 
                               'Num':f'Copy 1 ;ADME PK;Mean;Cl_obs(mL/min/kg);Dose: {dose_Cl};{admin_Cl};Species: {Species};(Num)'}
        Clobs_IV = extractPropertyDataFromD360Table(row, colName_dict_Cl['Mod'], colName_dict_Cl['Num'])
        EstFa = calc_EstFa(PKF_PO, Clobs_IV)
        output = pd.Series([PKF_PO, EstFa])
    return output