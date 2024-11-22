#!/bin/bash -l

date_str=$(date +"%Y%b%d")
rng="666666"
tmpfolder='tmp_folder_regression'
##########################################################
file_in='./Kymera_ADME_PK_ALL_pull.csv'
del=','

col_ID='Compound Name'
colSmi='Smiles'
##########################################################
spt='temporal'    #'random', 'butina'
colDate="Created On"
##########################################################
d_fps="False"
d_rd="True"
rd_norm="True"
d_cx="False"
cx_norm="False"

##########################################################
CV="10"
hasVal="True"
HPT="True"
##########################################################
colAssay='ADME AlphaLogD;Mean;AlphaLogD;(Num)'
model_Type="regression"
yTransform="log10"
##
# colAssay='low_logD'
# model_Type="classification"
# yTransform="One-hot"
##########################################################
model_rf="True"
model_linear="False"
model_svm="False"
model_mlp="False"
model_knn="False"
model_knnk="3"
model_xgb="False"
##########################################################
python DIYML.py -i "$file_in" -d "$del" --colId "$col_ID" --colSmi "$colSmi" --colAssay "$colAssay" --yTransform "$yTransform" --colDate "$colDate" --tmpFolder "$tmpfolder" --desc_fps "$d_fps" --desc_rdkit "$d_rd" --desc_rd_norm "$rd_norm" --desc_cx "$d_cx" --desc_cx_norm "$cx_norm" --split "$spt" --CV "$CV" --hasVal "$hasVal" --randomSeed "$rng" --modelType "$model_Type" --model_rf "$model_rf" --model_linear "$model_linear" --model_svm "$model_svm" --model_mlp "$model_mlp" --model_knn "$model_knn" --model_knnk "$model_knnk" --model_xgb "$model_xgb" --HPT "$HPT"


