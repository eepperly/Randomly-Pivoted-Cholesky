#!/bin/sh

# RLS samplers
if [ ! -f "recursive_nystrom.py" ] 
then
    wget https://raw.githubusercontent.com/axelv/recursive-nystrom/master/recursive_nystrom.py
fi

if [ ! -f "bless.py" ] 
then
    wget https://raw.githubusercontent.com/LCSL/bless/master/bless.py
fi

# Python packages
pip install qml
pip install dppy

# QM9 dataset
if [ ! -d "molecules" ] 
then
    wget https://figshare.com/ndownloader/files/3195389
    mkdir -p molecules
    cd molecules
    mv ../3195389 .
    tar -xvf 3195389
fi

# Data folder
mkdir -p data
