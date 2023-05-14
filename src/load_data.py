import argparse
import gzip
import os
import sys
from urllib.request import urlretrieve

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

avai_dataset = [
    "iris",
    "mnist",
    "fashion_mnist",
]


def load_data(dataset_name):
    if dataset_name in ["fashion_mnist", "mnist"]:
        Xtrain, ytrain, Xtest, ytest = load_data_mnist(dataset_name)
        X = np.concatenate((Xtrain, Xtest), axis=0)
        y = np.concatenate((ytrain, ytest), axis=0)

    if dataset_name == "iris":
        iris = load_iris()
        X = iris.data
        y = iris.target

    if dataset_name == "digits":
        digits = load_digits()
        X = digits.data
        y = digits.target

    if dataset_name == "wiscosin":

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        data = pd.read_csv(url, header=None)

        X = data.iloc[:, 2:].to_numpy(
        )  # exclude first two columns (id and diagnosis)
        y = (data.iloc[:, 1].map({
            "M": 1,
            "B": 0
        }).to_numpy())  # convert diagnosis column to binary labels

    if dataset_name == "wine":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        data = pd.read_csv(url, header=None)

        X = data.iloc[:, 1:].to_numpy()
        y = data.iloc[:, 0].to_numpy()

    if dataset_name == "seeds":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        raw_data = urllib.request.urlopen(url)
        dataset = np.loadtxt(raw_data)

        X = dataset[:, :-1]
        y = dataset[:, -1]

    if dataset_name == "ionosphere":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        data = pd.read_csv(url, header=None)
        data.iloc[:, -1] = pd.Categorical(data.iloc[:, -1]).codes

        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    if dataset_name == "yeast":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        data = pd.read_csv(url, delim_whitespace=True, header=None)
        X = data.iloc[:, 1:9].to_numpy()
        y = pd.Categorical(data.iloc[:, -1]).codes

    if dataset_name == "new_thyroid":
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
        data = pd.read_csv(url, header=None)

        X = data.iloc[
            1:,
        ].to_numpy()
        y = data.iloc[:, 0].to_numpy()
    return X, y


def create_split_indices(dataset_name):
    #for small data set, create split and return index corresponding with merge
    if dataset_name in ["fashion_mnist", "mnist"]:
        Xtrain, ytrain, Xtest, ytest = load_data_mnist(dataset_name)
        train_indices = range(0, Xtrain.shape[0])
        test_indices = range(Xtrain.shape[0], Xtrain.shape[0] + Xtest.shape[0])

    else:
        merged_data, y = load_data(dataset_name)

        Xtrain, Xtest, train_indices, test_indices = train_test_split(\
            merged_data, np.arange(merged_data.shape[0]), test_size=0.2, random_state=42)
        ytrain = y[train_indices]
        ytest = y[test_indices]

    return Xtrain, Xtest, ytrain, ytest, train_indices, test_indices


def load_data_mnist(dataset_name):
    """
    Args:
    - dataset_name: ['fashion_mnist', 'mnist']
    Returns:
    - (Xtrain, ytrain, Xtest, ytest)

    """

    dataset_names = ["fashion_mnist", "mnist"]
    if dataset_name not in dataset_names:
        raise ValueError("Invalid dataset_name, Expected one of :%s " %
                         dataset_names)
    urls = {
        "mnist":
            "http://yann.lecun.com/exdb/mnist/",
        "fashion_mnist":
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    }

    RESOURCES = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    if os.path.isdir("data") == 0:
        os.mkdir("data")
    if os.path.isdir("data/raw/{}".format(dataset_name)) == 0:
        os.mkdir("data/raw/{}".format(dataset_name))
    data_path = "data/raw/{}/raw/".format(dataset_name)
    if os.path.isdir(data_path) == 0:
        os.mkdir(data_path)
    #print(data_path)

    for name in RESOURCES:
        #print(data_path + name)
        if os.path.isfile(data_path + name) == 0:
            url = urls.get(dataset_name) + name
            urlretrieve(url, os.path.join(data_path, name))
            print("Downloaded %s to %s" % (name, data_path))

    Xtrain, ytrain = load_mnist(data_path, kind="train")
    Xtest, ytest = load_mnist(data_path, kind="t10k")

    return Xtrain, ytrain, Xtest, ytest


def load_mnist(path, kind="train"):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# sys.path.append("..")

# breast_tissue = read_excel_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls')
# breast_tissue = subset(breast_tissue, select = -c(1) )
# parkinsons = read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')
# parkinsons = subset(parkinsons, select = -c(name) )
# new_thyroid = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data', header=F)
# breast_cancer_wisconsin = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = F)
# breast_cancer_wisconsin = subset(breast_cancer_wisconsin, select = -c(1) )
# letter = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data', header=F)
# spam = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=F )
# yeast = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', header=F, sep="")
# yeast = subset(yeast, select = -c(1) )
# lymphography = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data', header=F)
# lymphography
# glass = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', header = F)
# glass = subset(glass, select = -c(1) )
# segmentation = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data', skip = 4, header = F)
# wisconsin = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=F)
# wisconsin = subset(wisconsin, select = -c(1) )
# soybean = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data", header=F)
# wine_quality = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', header=T, sep = ";")
#
# ecoli = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', header=F, sep="")
# ecoli  = subset(ecoli, select = -c(1))
