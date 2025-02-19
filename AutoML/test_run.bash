#!/bin/bash -l

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"

# ## ------------------ get a running copy of the update file ------------------
# ImgDir="./Update/"
# JobDir="./Update_$dateToday/"
# cp -r $ImgDir $JobDir
# echo "Copy update files from $ImgDir to $JobDir"
# cd $JobDir

## ------------------ run the command ------------------
# bash2py="./bash2py_yjing_local.bash"
bash2py="/mnt/data0/Research/0_Test/cx_pKa/bash2py_yjing_local.bash"
JobDir='./'

## step-1
Dir_step1="$JobDir/1_DataPrep"
fileInS1="$Dir_step1/0_Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
fileOutS1="$Dir_step1/results/data_input_clean.csv"

colId='Compound Name'
colSmi='Structure'
colDate="ADME MDCK(WT) Permeability;Concat;Run Date"
colAssay='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
colAssayMod='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'
colPreCalcDesc="ADME MDCK(WT) Permeability;Mean;A to B Recovery (%),ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Num)"

$bash2py python "$Dir_step1"/DataPrep.py -i "$fileInS1" -d ',' --detectEncoding --colId "$colId" --colSmi "$colSmi" --colAssay "$colAssay" --colAssayMod "$colAssayMod" --colPreCalcDesc "$colPreCalcDesc" -o "$fileOutS1"


## step-2
Dir_step2="$JobDir/1_DataPrep"
split='temporal'      #'random', 'diverse'
CV=10
rng=666666
hasVal=True
folderOutS2="$Dir_step2/results"
$bash2py python "$Dir_step2"/Data_Split.py -i "$fileOutS1" -d ',' --colId "$colId" --colSmi "$colSmi" --colDate "$colDate" --split "$split" --CV "$CV" --rng "$rng" --hasVal "$hasVal" -o "$folderOutS2"

## step-3
Dir_step3="$JobDir/2_FeatPrep/DescGen"
folderOutS3="$Dir_step3/results"
$bash2py python "$Dir_step3"/DescGen.py -i "$fileOutS1" -d ',' --colId "$colId" --colSmi "$colSmi" --desc_rdkit --desc_fps --desc_cx -o "$folderOutS3"

## step-4




