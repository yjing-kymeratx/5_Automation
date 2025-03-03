#!/bin/bash -l
## ===========================================
## ============== Python SetUp ===============
## ===========================================

## 1) rm the existing global variable for python
unset PYTHONPATH
export PATH=/fsx/home/yjing/apps/anaconda3/envs/yjing/bin/:$PATH
#export PATH=/fsx/home/yjing/apps/chemaxon/marvinsuite/bin/:$PATH
export PATH=/mnt/jchem/bin/:$PATH
export SCHRODINGER=/fsx/home/yjing/apps/schrodinger2023-2
# export PYTHONPATH=$PYTHONPATH:/fsx/home/yjing/apps/anaconda3/envs/yjing/bin/python

## 2) Initiate conda for python env
# source /fsx/home/yjing/apps/anaconda3/etc/profile.d/conda.sh

## 3) Activate custom conda environments
# conda activate /fsx/home/yjing/apps/anaconda3/envs/yjing    ## 'gpb39' ==> 'yjing'

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
