#!/bin/bash -l
# Make_Predictions.bash model.pickle molecules.sdf 

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"
########################################################
# ## ------------------ get a running copy of the update file ------------------
########################################################
# ImgDir="./Update/"
# JobDir="./Update_$dateToday/"
# cp -r $ImgDir $JobDir
# echo "Copy update files from $ImgDir to $JobDir"
# cd $JobDir

########################################################
## ------------------ run the command ------------------
########################################################
# bash2py="./bash2py_yjing_local.bash"
bash2py="/mnt/data0/Research/0_Test/cx_pKa/bash2py_yjing_local.bash"
modelDir="/mnt/data0/Research/5_Automation/AutoML"
JobDir='./'
resultDir="$JobDir"

# fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export_top300.csv"
iType="csv"
sep=','
colId='Compound Name'
colSmi='Structure'
colAssay='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'

modelFile="$modelDir/Results/Models/Best_AutoML_models.pickle"
fileOut="$resultDir/results.csv"

$bash2py python "$JobDir"/ModelPrediction.py -i "$fileIn" --inputType "$iType" -d "$sep" --colId "$colId" --colSmi "$colSmi" --colAssay "$colAssay" --modelFile "$modelFile" -o "$fileOut"


