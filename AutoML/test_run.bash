#!/bin/bash -l

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
JobDir='./'
resultDir="$JobDir/results"

## ------------------ step-1 csv loader & data clean ------------------
# fileInS1="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
fileInS1="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export_top40.csv"
colId='Compound Name'
colSmi='Structure'
colAssay='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
colAssayMod='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'
fileOutS1="$resultDir/data_input_clean.csv"

$bash2py python "$JobDir"/DataPrep.py -i "$fileInS1" -d ',' --detectEncoding --colId "$colId" --colSmi "$colSmi" --colAssay "$colAssay" --colAssayMod "$colAssayMod" -o "$fileOutS1"

## ------------------ step-2 train-val-test split ------------------
fileInS2="$fileOutS1"
colId="$colId"
colSmi="$colSmi"
colDate="ADME MDCK(WT) Permeability;Concat;Run Date"
colPreCalcDesc="ADME MDCK(WT) Permeability;Mean;A to B Recovery (%),ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Num)"
split='random'     #'random', 'diverse', 'temporal'
CV=10
rng=666666
hasVal=True
folderOutS2="$resultDir"

$bash2py python "$JobDir"/DataSplit.py -i "$fileInS2" -d ',' --colId "$colId" --colSmi "$colSmi" --colDate "$colDate" --split "$split" --CV "$CV" --rng "$rng" --hasVal "$hasVal" -o "$folderOutS2"

## ------------------ step-3 generate descriptors ------------------
fileInS3="$fileOutS1"
colId="$colId"
colSmi="$colSmi"
desc_rdkit=True
desc_fps=True
desc_cx=True
norm=True
imput=True
folderOutS3="$resultDir"

$bash2py python "$JobDir"/DescGen.py -i "$fileInS3" -d ',' --colId "$colId" --colSmi "$colSmi" --desc_rdkit "$desc_rdkit" --desc_fps "$desc_fps" --desc_cx "$desc_cx" --colPreCalcDesc "$colPreCalcDesc" --norm "$norm" --imput "$imput" -o "$folderOutS3"

## ------------------ step-4 feature selection ------------------
# folderInS4="$resultDir"
# desc_rdkit=True
# desc_fps=True
# desc_cx=True
# $bash2py python "$JobDir"/FeatSele.py -i "$folderInS4" -d ',' --desc_rdkit "$desc_rdkit" --desc_fps "$desc_fps" --desc_cx "$desc_cx" --colId "$colId" -o "$folderOutS3"





