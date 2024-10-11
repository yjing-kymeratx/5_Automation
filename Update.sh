#!/bin/bash -l

## git add all files
git add .
git add -A

## Get today's date and save it in a string variable
date_str=$(date +"%Y%b%d")

## Convert the month to short form and make the first letter uppercase
date_str=$(echo "$date_str" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
echo "Today's date is: $date_str"

## committee
git commit -m "$date_str"


## push to github
git push

#

#

#

#
