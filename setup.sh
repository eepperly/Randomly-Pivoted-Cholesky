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
pip3 install qml
pip3 install dppy

# QM9 dataset
cd experiments
if [ ! -d "molecules" ] 
then
    wget https://figshare.com/ndownloader/files/3195389
    mkdir -p molecules
    cd molecules
    mv ../3195389 .
    tar -xvf 3195389
    rm 3195389
    cd ..
fi

# Alanine dipeptide
wget http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-3x250ns-heavy-atom-positions.npz
wget http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz

# Data and figure folder
mkdir -p data
mkdir -p figs
cd ..
