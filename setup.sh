#!/bin/sh

# RLS samplers
wget https://raw.githubusercontent.com/axelv/recursive-nystrom/master/recursive_nystrom.py
wget https://raw.githubusercontent.com/LCSL/bless/master/bless.py

# Python packages
pip install qml
pip install dppy

# QM9 dataset
wget https://figshare.com/ndownloader/files/3195389
mkdir -p molecules
cd molecules
mv ../3195389 .
tar -xvf 3195389

