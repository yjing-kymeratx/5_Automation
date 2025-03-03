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
# bash2py="$JobDir/bash2py_yjing.bash"
bash2py="/mnt/data0/Research/0_Test/cx_pKa/bash2py_yjing_local.bash"

# fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export.csv"
fileIn="$JobDir/Data/DataView_MDCK_MDR1__Permeability_1__export_top500.csv"
fileParam="$JobDir/Data/Parameters.csv"

$bash2py python "$JobDir"/AutoML.py -i "$fileIn" -p "$fileParam"

