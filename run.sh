#!/bin/bash

# Before use the script, you must be able to connect to all hosts without a password using ssh.
# It can be setup by running 'ssh-keygen -t dsa; ssh-copy-id <hostname>' and using an empty passphrase.
# The predictors path and the target path must be in your current directory (pwd).
# Be sure the file 'transition.txt' doesn't exist before run this script.

# Parameters
X_dataset="X.txt" # Predictors path
Y_dataset="Y.txt" # Target path
outputName="out"
corrTh=0.3 # Correlation threshold
maxVL_OPS=10 # Max Num Latent Variables OPS
maxVL_model=3 # Max Num Latent Variables Model
maxVariables=5
nameInfoVec='' # Possible values: 'reg', 'corr' or 'prod'. If you want to run the OPS_auto (for all the informative vectors), let nameInfoVec=''
verbose=1 # Let verbose=1 if you want to show the progress or let verbose=0 otherwise.

# The variable 'servers' contains the names of all hosts which will be used. ':' is the local host. The number before the '/' is the number of cores used in that host.
# To add more hosts, write each host name separating with a comma. For example: servers='8/:,4/host1,8/host2'.
servers='8/:'

# Max number of process running in parallel
maxProcess=8

# File to save the results from different processes. This file will be appended during the process.
touch transition.txt

if [ "$nameInfoVec" == "corr" ] 
then
	maxVL_OPS=1
fi

if [ $verbose == 1 ]
then
	parallel --progress -j$maxProcess -S $servers --workdir $(pwd) $(which python3) main_crossVal1.py $X_dataset $Y_dataset $corrTh {1} $maxVL_model $maxVariables $nameInfoVec ::: $(seq 1 $maxVL_OPS)
else
	parallel -j$maxProcess -S $servers --workdir $(pwd) $(which python3) main_crossVal1.py $X_dataset $Y_dataset $corrTh {1} $maxVL_model $maxVariables $nameInfoVec ::: $(seq 1 $maxVL_OPS)	
fi

$(which python3) main_crossVal2.py $X_dataset $Y_dataset $outputName $corrTh $nameInfoVec

rm transition.txt
