#!/bin/sh

wget https://raw.githubusercontent.com/axelv/recursive-nystrom/master/recursive_nystrom.py

pip install qml
pip install dppy

wget https://figshare.com/ndownloader/files/3195389
mkdir -p molecules
cd molecules
mv ../3195389 .
tar -xvf 3195389

