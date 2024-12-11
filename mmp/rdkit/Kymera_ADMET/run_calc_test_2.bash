#!/bin/bash -l

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"

## ------------------ get a running copy of the update file ------------------
RootDir="/mnt/data0/Research/5_Automation/mmp/rdkit/Kymera_ADMET"
ImgDir="$RootDir/Update"
# JobDir="$RootDir/Update_$dateToday"    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
JobDir="$RootDir/Update_2_$dateToday"    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo "1. Initiate job directory <$JobDir>"

## if job dir exist delete it
if [ -d "$JobDir" ]; then
    echo "    Folder <$JobDir> exists. Deleting it ..."
    rm -rf "$JobDir"
fi
##
cp -r $ImgDir $JobDir
echo "    Copy files from <$ImgDir> to <$JobDir>"

## ------------------ Check if a data file exists ------------------
echo "2. Check if a csv file already existed in job dir <$JobDir>"

for file in "$JobDir/D360_dataset_q_id$queryID"*; do
    if [ -f "$file" ]; then
        echo "    File $file exists. Deleting..."
        rm "$file"
    fi
done

## ------------------ Check if the results/temp folder exists ------------------
echo "3. Check if the <tmp> & <results> folder existed in job dir <$JobDir>"

resultsFolderPath="$JobDir/results"
if [ -d "$resultsFolderPath" ]; then
    echo "    <$resultsFolderPath> Folder exists. Deleting..."
    rm -rf "$resultsFolderPath"
fi
mkdir $resultsFolderPath
echo "    making a directory $resultsFolderPath"

## 
tmpFolderPath="$JobDir/tmp"
if [ -d "$tmpFolderPath" ]; then
    echo "    <$tmpFolderPath> Folder exists. Deleting..."
    rm -rf "$tmpFolderPath"
fi
mkdir $tmpFolderPath
echo "    making a directory $tmpFolderPath"


## ------------------ run the command ------------------
echo "4. Run commands in job dir <$JobDir>"

echo "    Change directory to $JobDir"
cd $JobDir

echo "    run commandline script ..."
bash2py="$JobDir/bash2py_mmp_local.bash"
pyScript="$JobDir/calc_MMPs.py"
queryID=3539
echo "--------------------------------------------"
# $bash2py python $pyScript -q $queryID    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
inFileName="$RootDir/Update_20241211/tmp/D360_dataset_q_id3539_111224_1528.csv"    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
$bash2py python $pyScript -i $inFileName    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo "--------------------------------------------"

## ------------------ after done with the job, move to "Completed" folder ------------------
CompleteDir="$RootDir/Completed"
# JobDirNew="$CompleteDir/Update_$dateToday"    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
JobDirNew="$CompleteDir/Update_2_$dateToday"    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo "5. Move the job dir to <Completed> folder <$JobDirNew>."

## if job dir exist delete it
if [ -d "$JobDirNew" ]; then
    echo "    Folder <$JobDirNew> exists. Deleting..."
    rm -rf "$JobDirNew"
fi
# mkdir $JobDirNew
# echo "making a directory $JobDirNew"

cd "$RootDir"
mv $JobDir $JobDirNew
echo "    move folder <$JobDir> to <$JobDirNew>"