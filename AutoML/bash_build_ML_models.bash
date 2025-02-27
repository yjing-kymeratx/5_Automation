#!/bin/bash -l
# Make_Predictions.bash input_data.csv 

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"
########################################################
# ## ------------------ get a running copy of the update file ------------------
########################################################
# ImgDir="./AutoML/"
# JobDir="./AutoML_$dateToday/"
# cp -r $ImgDir $JobDir
# echo "Copy update files from $ImgDir to $JobDir"
# cd $JobDir

########################################################
## ------------------ run the command ------------------
########################################################
# bash2py="./bash2py_yjing_local.bash"
bash2py="/mnt/data0/Research/0_Test/cx_pKa/bash2py_yjing_local.bash"
JobDir='./'
resultDir="$JobDir/Results"

## input 
fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export_top300.csv"
sep=","
colId='Compound Name'
colSmi='Structure'

colAssay='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
colAssayMod='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'
colDate="ADME MDCK(WT) Permeability;Concat;Run Date"    ## needed if 'temporal'

colPreCalcDesc="ADME MDCK(WT) Permeability;Mean;A to B Recovery (%),ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Num)"
colSplit="Split"

## S1
fileInS1=$fileIn
fileOutS1="$resultDir/data_input_clean.csv"
fileOutS1y="$resultDir/outcome_expt.csv"

## S2
fileInS2=$fileOutS1

split='diverse'     #'random', 'diverse', 'temporal'
CV=10
rng=666666
hasVal=True
fileOutS2="$resultDir/data_split_$split.csv"

## S3
fileInS3=$fileOutS1
desc_rdkit="True"
desc_fps="True"
desc_cx="True"
norm="True"
imput="True"

fileOutS3="$resultDir/descriptors_prep_merged.csv"

## S4
fileInS4_X=$fileOutS3
fileInS4_y=$fileOutS1y
fileInS4_S=$fileOutS2

modelType="regression"    ## "regression" or "classification"
nanFilter="True"
VFilter="True"
L2Filter="True"
FIFilter="True"

ImpuParamFile="$resultDir/feature_imputation_params.json"
folderOutS4="$resultDir/data_input_4_ModelBuilding.csv"

## S5
fileInS5=$folderOutS4
modelType=$modelType
ml_linear="True"
ml_rf="True"
ml_svm="True"
ml_mlp="True"
ml_knn="True"

rng=$rng
njobs=-1
logy="True"
# doHPT="True"
doHPT="False"
folderOutS5="$resultDir/performance_results.csv"

CalcParamFile="$resultDir/feature_calculator_param.json"
NormParamFile="$resultDir/feature_normalization_params.json"
ImpuParamFile=$ImpuParamFile

## ------------------ step-1 csv loader & data clean ------------------
$bash2py python "$JobDir"/DataPrep.py -i "$fileInS1" -d "$sep" --detectEncoding --colId "$colId" --colSmi "$colSmi" --colAssay "$colAssay" --colAssayMod "$colAssayMod" -o "$fileOutS1" --outputy "$fileOutS1y"

## ------------------ step-2 train-val-test split ------------------
$bash2py python "$JobDir"/DataSplit.py -i "$fileInS2" -d "$sep" --colId "$colId" --colSmi "$colSmi" --colDate "$colDate" --split "$split" --CV "$CV" --rng "$rng" --hasVal "$hasVal" --cols "$colSplit" -o "$fileOutS2"

## ------------------ step-3 generate descriptors ------------------
$bash2py python "$JobDir"/DescGen.py -i "$fileInS3" -d "$sep" --colId "$colId" --colSmi "$colSmi" --desc_rdkit "$desc_rdkit" --desc_fps "$desc_fps" --desc_cx "$desc_cx" --colPreCalcDesc "$colPreCalcDesc" --norm "$norm" --imput "$imput" -o "$fileOutS3"

## ------------------ step-4 feature selection ------------------
$bash2py python "$JobDir"/FeatSele.py -x "$fileInS4_X" -y "$fileInS4_y" -s "$fileInS4_S" -d "$sep" --colId "$colId" --coly "$colAssay" --cols "$colSplit" --modelType "$modelType" --MissingValueFilter "$nanFilter" --ImputationParamFile "$ImpuParamFile" --VarianceFilter "$VFilter" --L2Filter "$L2Filter" --FIFilter "$FIFilter" -o "$folderOutS4"

## ------------------ step-5 ML modeling ------------------
$bash2py python "$JobDir"/ModelBuilding_r.py -i "$fileInS5" -d "$sep" --colId "$colId" --coly "$colAssay" --cols "$colSplit" --linear "$ml_linear" --rf "$ml_rf" --svm "$ml_svm" --mlp "$ml_mlp" --knn "$ml_knn" --njobs "$njobs" --rng "$rng" --logy $logy --doHPT "$doHPT" --calcParamJson "$CalcParamFile" --normParamJson "$NormParamFile" --impuParamJson "$ImpuParamFile" -o "$folderOutS5"

## ------------------ step-6 to be added ------------------