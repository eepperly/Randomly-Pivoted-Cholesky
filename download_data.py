#!/usr/bin/env python3

import requests
from libsvm.svmutil import svm_read_problem
import openml
import numpy as np
from scipy.io import savemat
from scipy.sparse import csr_matrix
from tqdm import tqdm
import bz2
import lzma
import os
from sklearn.model_selection import train_test_split

# AUXILIARY FUNCTIONS

def download_file(url, directory, compression = None):
    print(f"Downloading {url}")
    filename = url.split('/')[-1]
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    filepath = directory + filename
    with open(filepath, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.close()    
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")    
    return decompress_file(filepath)  

def decompress_file(filepath):
    
    def decompress_with(libr, extension):
        newfilepath = filepath[:(-len(extension))]
        with libr.open(filepath) as f, open(newfilepath, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)
            print(f"Writing f{newfilepath}")
            f.close()
            fout.close()
            os.remove(filepath)
        return newfilepath
    
    if filepath.lower().endswith("bz2"):
        return decompress_with(bz2, ".bz2")
    elif filepath.lower().endswith("xz"):
        return decompress_with(lzma, ".xz")
    else:
        return filepath
            

# CONSTANTS

seed = 926
raw_data_directory = "data/raw/"
mat_data_directory = "data/preprocessed/"

# DATASETS
# We download a superset of the datasets we use in the experiments. We decided to drop some of these because
# their sample size was too small.

# If a dataset includes an url for testing, we use the two datasets, otherwise we split them randomly according
# to the test_proportion.
datasets_libsvm = {
    "a9a": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t",
    },
    "cadata": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata",
        "test_proportion": 0.2,
    },
    "cod-rna": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
    },
    "connect-4": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/connect-4",
        "test_proportion": 0.2,
    },
    "covtype.binary": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "test_proportion": 0.2,
    },
     "ijcnn1": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
    },
    "phishing" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
        "test_proportion": 0.2,
    },  
    "sensit_vehicle" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2",
    },
    "sensorless": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale.tr",
    },
    "skin_nonskin" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin",
        "test_proportion": 0.2,
    },
    "YearPredictionMSD" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
        "test" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2",
    },
    "w8a": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t"
    },
}

# If the test_proportion is provided, an openml dataset is split randomly according to the test_proportion,
# otherwise the first ntr points are used for training and the subsequent nts data points for testing.
datasets_openml = {
    "ACSIncome": {
        "id": 43141,
        "target": "PINCP",
        "test_proportion": 0.2, 
    },
    "Airlines_DepDelay_1M": {
        "id": 42721,
        "target": "DepDelay",
        "test_proportion": 0.2,
    },
    "Click_prediction_small": {
        "id": 1218,
        "target": "click",
        "test_proportion": 0.2,
    },
    "COMET_MC_SAMPLE": {
        "id": 23395,
        "target": "label",
        "test_proportion": 0.2,
    },
    "creditcard": {
        "id": 1597,
        "target" : "Class",
        "test_proportion": 0.2,
    },
    "diamonds": {
        "id": 42225,
        "target": "price",
        "test_proportion": 0.2,
    },
    "HIGGS" : {
        "id": 42769,
        "target": "target",
        "ntr": 500000,
        "nts": 500000,
    },
    "hls4ml_lhc_jets_hlf": {
        "id": 42468,
        "target": "class",
        "test_proportion": 0.2
    },
    "jannis": {
        "id": 44131,
        "target": "class",
        "test_proportion": 0.2,
    },
    "Medical-Appointment": {
        "id": 43617,
        "target": "show",
        "test_proportion": 0.2,
    },
    "MiniBoonNE": {
        "id": 41150,
        "target": "signal",
        "test_proportion": 0.2,
    },
    "MNIST": {
        "id": 554,
        "target": "class",
        "ntr": 60000,
        "nts": 10000,
    },
    "santander": {
        "id": 42435,
        "target": "target",
        "test_proportion": 0.2,
        "remove_features": ["ID_code"],
    },
    "volkert": {
       "id": 41166,
        "target": "class",
        "test_proportion": 0.2,         
    },
    "yolanda": {
        "id": 42705,
        "target": "101",
        "test_proportion": 0.2, 
    },
}


# Downloads datasets from LIBSVM
for k, data in datasets_libsvm.items():
    url = data["train"]
    dataset_path = download_file(url, raw_data_directory)
    Ytr, Xtr = svm_read_problem(dataset_path, return_scipy = True)
    
    if "test" in data:
        url_ts = data["test"]
        dataset_path_ts = download_file(url_ts, raw_data_directory)
        Yts, Xts = svm_read_problem(dataset_path_ts, return_scipy = True)
    # Take a random split for train and test
    elif "test_proportion" in data:
        Xtr, Xts, Ytr, Yts = train_test_split(Xtr, Ytr, test_size=data["test_proportion"], random_state=seed)

    # Use the first ntr points for training and the next nts points for testing.    
    else:
        Ytraux = Ytr[:(data["ntr"])]
        Xtraux = Xtr[:(data["ntr"])]
        Yts = Ytr[data["ntr"]:(data["ntr"] + data["nts"])]
        Xts = Xtr[data["ntr"]:(data["ntr"] + data["nts"])]
        Ytr = Ytraux
        Xtr = Xtraux
        
    datadir = {'Xtr': Xtr , 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts}
    savemat(mat_data_directory + k +".mat", datadir)
    

# Downloads datasets from OpenML
for k, data in datasets_openml.items():
    print(f"Downloading {k}")
    dataset = openml.datasets.get_dataset(data["id"])
    X, y, _, attr = dataset.get_data(dataset_format="array")

    # The target is encoded in the features
    if "target" in data and y is None:
        idx = np.where(np.array(attr) == data["target"])[0][0]
        y = X[:, idx]
        X = np.delete(X, idx, 1) # Deletes the (idx)th column
        
    elif y is None:
        raise Exception(f"Error with the data structure defining datsets, field target needs to be set for dataset {k}.")

    # Remove unwanted features, e.g., IDs.     
    if "remove_features" in data:
        for feature in data["remove_features"]:
            idx = np.where(np.array(attr) == data["remove_features"])[0][0]
            X = np.delete(X, idx, 1)

    if "test_proportion" in data:        
        Xtr, Xts, Ytr, Yts = train_test_split(X, y, test_size=data["test_proportion"], random_state=seed)
        
    elif "ntr" in data and "nts" in data:
        Ytraux = y[:(data["ntr"])]
        Xtraux = X[:(data["ntr"])]
        Yts = y[data["ntr"]:(data["ntr"] + data["nts"])]
        Xts = X[data["ntr"]:(data["ntr"] + data["nts"])]
        Ytr = Ytraux
        Xtr = Xtraux
        
    else:
        raise Exception(f"Error with the data structure defining datsets, a (train, test) strategy needs to be provided for {k}")

    # OpenML likes single precision, we don't.
    Xtr = np.array(Xtr, dtype = np.float64)
    Ytr = np.array(Ytr, dtype = np.float64)
    Xts = np.array(Xts, dtype = np.float64)
    Yts = np.array(Yts, dtype = np.float64)
    
    datadir = {'Xtr': Xtr , 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts}
    savemat(mat_data_directory + k +".mat", datadir)   
