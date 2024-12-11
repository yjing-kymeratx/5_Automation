#!/bin/bash -l

## ------------------ get date today ------------------
dateToday=$(date +'%Y%m%d')
echo "Today is $dateToday"

## ------------------ get a running copy of the update file ------------------
ImgDir="./Update/"
JobDir="./Update_$dateToday/"
cp -r $ImgDir $JobDir
echo "Copy update files from $ImgDir to $JobDir"
cd $JobDir


## ------------------------------------
queryID=3539

## ------------------ Check if the data file exists ------------------
data_dir_path="./"

for file in "$data_dir_path"/D360_dataset_q_id"$queryID"*; do
    if [ -f "$file" ]; then
        echo "File $file exists. Deleting..."
        rm "$file"
    fi
done

## ------------------ Check if the results folder exists ------------------
results_folder_path="./results"

if [ -d "$results_folder_path" ]; then
    echo "<./results/> Folder exists. Deleting..."
    rm -rf "$results_folder_path"
fi

## ------------------ Check if the temp folder exists ------------------
tmp_folder_path="./tmp"

if [ -d "$tmp_folder_path" ]; then
    echo "<./tmp/> Folder exists. Deleting..."
    rm -rf "$tmp_folder_path"
fi


# ## ------------------ Check if the results pickle file exists ------------------
# results_file_path="./tmp/results_dict_tmp.pickle"

# if [ -f "$results_file_path" ]; then
#     echo "<results_dict_tmp.pickle> File exists. Deleting..."
#     rm "$results_file_path"
# fi

## ------------------ run the command ------------------
bash2py="./bash2py_mmp_local.bash"
$bash2py python cxAPI_runner.py -q $queryID    # -i DataView_Automation_WeeklyUpdate_1__export.csv

## 
cd "../"
JobDirNew="./Completed/Update_$dateToday/"
mv $JobDir $JobDirNew
