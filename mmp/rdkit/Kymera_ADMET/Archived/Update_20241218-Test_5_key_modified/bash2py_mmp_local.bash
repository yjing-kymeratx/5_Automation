#!/bin/bash -l



## ===========================================
## ============== Python SetUp ===============
## ===========================================

## 1) rm the existing global variable for python
unset PYTHONPATH

## 2) Initiate conda for python env
#source /home/yjing/anaconda3/etc/profile.d/conda.sh
# source /home/yjing/data0/software/anaconda/anaconda3/etc/profile.d/conda.sh


## 3) Activate custom conda environments
# conda activate /home/yjing/data0/software/anaconda/anaconda3/envs/ml    ## 'ml', 'gpb39' ==> 'yjing'
export PATH=/mnt/data0/software/anaconda/anaconda3/envs/mmp/bin/:$PATH

## 4) define global variables & import python packages?
# export PatGlobalVars=/fsx/bin/PatGlobalVars.py
# export yjingGlobalVars=/fsx/bin/yjingGlobalVars.py 

#echo "You provided command line arguments:" "$@"
#echo "You provided $# command line arguments"
myArr=( "$@" )    ## put all the imported parameters in the list myArr
myCmd=( "${myArr[@]:0:1}" )
myArg=( "${myArr[@]:1}" )
if [ -t 0 ]; then
   #  nothing coming from stdin
   #echo "Run the following: $@"
   "${myCmd[@]}" "${myArg[@]}"
else
   # stuff is coming from stdin
   #echo "Run the following: cat - | $@"
   cat - | "${myCmd[@]}" "${myArg[@]}"
fi
