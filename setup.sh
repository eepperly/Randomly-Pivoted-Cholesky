#!/bin/sh

# Run this script from the root directory of the repository
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Download a file with either wget or curl
fetch () { # usage: fetch URL OUTPUT_FILE
    if command -v wget > /dev/null
    then
        wget -O "$2" "$1"
    else
        curl -L -o "$2" "$1"
    fi
}

# RLS samplers
if [ ! -f "recursive_nystrom.py" ] 
then
    fetch https://raw.githubusercontent.com/axelv/recursive-nystrom/master/recursive_nystrom.py recursive_nystrom.py
fi

if [ ! -f "bless.py" ] 
then
    fetch https://raw.githubusercontent.com/LCSL/bless/master/bless.py bless.py
fi

# Python packages
pip3 install -r requirements.txt

# Data and figure folders
mkdir -p data
mkdir -p experiments/data experiments/figs
mkdir -p block_experiments/data block_experiments/figs

# QM9 dataset
cd "$ROOT/experiments"
if [ ! -d "molecules" ] 
then
    fetch https://figshare.com/ndownloader/files/3195389 3195389
    mkdir -p molecules
    tar -xf 3195389 -C molecules
    rm 3195389
fi

# Alanine dipeptide
for f in alanine-dipeptide-3x250ns-heavy-atom-positions.npz alanine-dipeptide-3x250ns-backbone-dihedrals.npz
do
    if [ ! -f "$f" ]
    then
        fetch http://ftp.imp.fu-berlin.de/pub/cmb-data/$f $f
    fi
done

# Datasets for the experiments on many matrices
# (stored in data/preprocessed/ and used by experiments/many_matrices.py
# and block_experiments/performance.py)
cd "$ROOT"
python3 download_data.py
