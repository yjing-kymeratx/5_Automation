#!/bin/bash -l

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a new directory with the name using current data
if [ ! -d "./Archived/$current_date" ]; then
    # If it doesn't exist, create it
    mkdir "./Archived/$current_date"
fi


# If data file exists, move it to the new directory
data_file="Kymera.tpdecomp.data.csv"
if [ -f "$data_file" ]; then
    mv $"$data_file" "./Archived/$current_date"

# scp the data file to date folder from fsx
scp yjing@Cluster_cpu:"/fsx/data/AUTOMATION/DATA/$data_file" ./
fi