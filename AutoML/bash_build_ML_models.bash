#!/bin/bash -l
# Make_Predictions.bash input_data.csv 

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"
random_number=$RANDOM
########################################################
# ## ------------------ get a running copy of the update file ------------------
########################################################
# ImgDir="/fsx/home/yjing/models/AutoML"
# RootDir="/fsx/home/yjing/models/Test"
# JobDir="$RootDir/AutoML_""$dateToday""_""$RANDOM"
# cp -r $ImgDir $JobDir
# echo "Copy update files from $ImgDir to $JobDir"
# cd $JobDir
# echo "Changed directory to $JobDir"
JobDir='./'

########################################################
## ------------------ run the command ------------------
########################################################
# bash2py="/fsx/home/yjing/models/DIYML/bash2py_yjing.bash"
# bash2py="/fsx/home/yjing/models/AutoML/bash2py_yjing.bash"
bash2py="$JobDir/bash2py_yjing.bash"

## S1: input 
# fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export_top50.csv"
fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
sep=","
colId='Compound Name'
colSmi='Structure'
colAssay='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Num)'
colAssayMod='ADME MDCK(WT) Permeability;Mean;A to B Papp (10^-6 cm/s);(Mod)'
colDate="ADME MDCK(WT) Permeability;Concat;Run Date"    ## needed if 'temporal'
colPreCalcDesc="ADME MDCK(WT) Permeability;Mean;A to B Recovery (%),ADME MDCK(WT) Permeability;Mean;B to A Papp (10^-6 cm/s);(Num)"

## S2: split params
split='diverse'     #'random', 'diverse', 'temporal'
colSplit="Split"
CV="10"
hasVal="True"

## S3: calc descriptors
desc_rdkit="True"
desc_fps="True"
desc_cx="True"
norm="True"
imput="True"

## S4&5: feature sele & model building
modelType="regression"    ## "regression" or "classification"
nanFilter="True"
VFilter="True"
L2Filter="True"
FIFilter="True"

ml_linear="True"
ml_rf="True"
ml_svm="True"
ml_mlp="True"
ml_knn="True"
doHPT="True"
# doHPT="False"

## sys configs
rng=666666
njobs=-1
logy="True"

## output
resultDir="$JobDir/Results"

tmpFile_1_clean="$resultDir/data_input_clean.csv"
tmpFile_1_y="$resultDir/outcome_expt.csv"
tmpFile_2_split="$resultDir/data_split_$split.csv"
tmpFile_3_desc="$resultDir/descriptors_prep_merged.csv"
tmpFile_4_all4ml="$resultDir/data_input_4_ModelBuilding.csv"
tmpFile_5_modelPerfm="$resultDir/performance_results.csv"

tmpFile_CalcParam="$resultDir/feature_calculator_param.json"
tmpFile_NormParam="$resultDir/feature_normalization_params.json"
tmpFile_ImpuParam="$resultDir/feature_imputation_params.json"



## ------------------ step-1 csv loader & data clean ------------------
$bash2py python "$JobDir"/DataPrep.py -i "$fileIn" -d "$sep" --detectEncoding --colId "$colId" --colSmi "$colSmi" --colAssay "$colAssay" --colAssayMod "$colAssayMod" -o "$tmpFile_1_clean" --outputy "$tmpFile_1_y"

## ------------------ step-2 train-val-test split ------------------
$bash2py python "$JobDir"/DataSplit.py -i "$tmpFile_1_clean" --colId "$colId" --colSmi "$colSmi" --colDate "$colDate" --split "$split" --CV "$CV" --rng "$rng" --hasVal "$hasVal" --cols "$colSplit" -o "$tmpFile_2_split"

## ------------------ step-3 generate descriptors ------------------
$bash2py python "$JobDir"/DescGen.py -i "$tmpFile_1_clean" --colId "$colId" --colSmi "$colSmi" --desc_rdkit "$desc_rdkit" --desc_fps "$desc_fps" --desc_cx "$desc_cx" --colPreCalcDesc "$colPreCalcDesc" --norm "$norm" --imput "$imput" -o "$tmpFile_3_desc"

## ------------------ step-4 feature selection ------------------
$bash2py python "$JobDir"/FeatSele.py -x "$tmpFile_3_desc" -y "$tmpFile_1_y" -s "$tmpFile_2_split" --colId "$colId" --coly "$colAssay" --cols "$colSplit" --modelType "$modelType" --MissingValueFilter "$nanFilter" --impuParamJson "$tmpFile_ImpuParam" --VarianceFilter "$VFilter" --L2Filter "$L2Filter" --FIFilter "$FIFilter" -o "$tmpFile_4_all4ml"

## ------------------ step-5 ML modeling ------------------
$bash2py python "$JobDir"/ModelBuilding.py -i "$tmpFile_4_all4ml" --colId "$colId" --coly "$colAssay" --cols "$colSplit" --linear "$ml_linear" --rf "$ml_rf" --svm "$ml_svm" --mlp "$ml_mlp" --knn "$ml_knn" --njobs "$njobs" --rng "$rng" --logy $logy --doHPT "$doHPT"  --rng "$rng" --calcParamJson "$tmpFile_CalcParam" --normParamJson "$tmpFile_NormParam" --impuParamJson "$tmpFile_ImpuParam" -o "$tmpFile_5_modelPerfm"

## ------------------ step-6 to be added ------------------