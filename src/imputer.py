import os
import sys

import sklearn.neighbors._base  # scikit-learn==1.1.2
from fancyimpute import SoftImpute  # install  then install cvxopt-1.2.6
# import softimpute
# sys.path.append("")
# this is for the package missingpy

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base

from missingpy import MissForest

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# from src.PCA.missMDA import *
import time
import numpy as np

# import to run R code : if got error makee sure : conda install libcblas
import pandas as pd

# import rpy2.rinterface_lib.callbacks
# my_callback = lambda *args: None
# rpy2.rinterface_lib.callbacks.consolewrite_warnerror =  my_callback
#
# if getting error importing packages; make sure you done this
# export RPY2_CFFI_MODE=API

# gain

# from src.GAIN.gain import gain

# TODO
# [x] mean
# [x] softimpute (fancyimpute)
# [x] mice (sklearn)
# [x] imputePCA (R)
# [x] missingpy - MissForest: tunning param
# [x] EM (R)
# [x] KNN (sklearn)
# [] ginn
# [] gain
# [] dimv
# [] pipeline with Makefile; addding try catch if running multilple algorithm
# [] docker to help install


def mean_imputer(X, **kwargs):
    start = time.time()
    imputer = SimpleImputer(**kwargs)

    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start
    return Ximp, duration


def softimpute_imputer(X, **kwargs):
    start = time.time()

    imputer = SoftImpute(**kwargs)  # default maxit = 1000
    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start

    return Ximp, duration


def mice_imputer(X, **kwargs):
    start = time.time()
    imputer = IterativeImputer(**kwargs)  # max_iter=10, random_state=0)
    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start

    return Ximp, duration


def imputepca_imputer(X, **kwargs):
    # this code is for not prining R console
    import logging
    import rpy2.rinterface_lib.callbacks

    rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()

    args_string = ""
    for key, value in kwargs.items():
        args_string += f"{key}={str(value).lower()}, "
    args_string = args_string[:-2]  # Remove the trailing comma and space

    # import rpy2.robjects as robjects
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    rcode_string = """
        install.packages("remotes")
        library(remotes)
        install_github("cran/missMDA") 
        library("missMDA")
        print("complete install")
        print(dim(data))
        imputing <- function(data){{
            start <- Sys.time()

            #ncomp = missMDA::estim_ncpPCA(data, ncp.min=2)
            #print(paste0("number of component choosen: ", ncomp$ncp))

            fit_train= missMDA::imputePCA(data, {})

            imp = fit_train$completeObs[,,drop=F]

            end <- Sys.time()
            duration <- end - start  

            return(list("imp" = imp, "duration" = duration))
            }}
        """.format(
        args_string
    )

    Ximp = X.copy()
    mmask = np.isnan(X)

    missMDA = SignatureTranslatedAnonymousPackage(rcode_string, "missMDA")

    # imputePCA can not handle std = 0
    stds_filter = np.nanstd(X, axis=0) != 0
    print(np.sum(stds_filter))

    indices = np.arange(X.shape[1])
    no0std_indices = indices[stds_filter]
    X_no0std = X[:, no0std_indices]

    # start impute
    result = missMDA.imputing(pd.DataFrame(X_no0std))

    imp_no0std = result.rx2("imp")
    duration = result.rx2("duration")

    # fill to the original matrix
    X_no0std[np.isnan(X_no0std)] = imp_no0std[np.isnan(X_no0std)]
    X[:, no0std_indices] = X_no0std
    # if value missing lie in the column have std = 0-> then fill with the others value in that column (or mean)
    print(sum(np.isnan(X)))
    imputer = SimpleImputer(**kwargs)
    XImpMean = imputer.fit_transform(X)

    Ximp[np.isnan(X)] = XImpMean[np.isnan(X)]

    return Ximp, duration


def em_imputer(X, **kwargs):  # 70p -  1 iteration
    # this code is for not prining R console
    import logging
    import rpy2.rinterface_lib.callbacks

    rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    # import rpy2.robjects as robjects
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    args_string = ""
    for key, value in kwargs.items():
        args_string += f"{key}={str(value).lower()}, "
    args_string = args_string[:-2]  # Remove the trailing comma and space

    rcode_string = """
        install.packages("missMethods")
        library("missMethods")
        imputing<- function(data){{
            start <- Sys.time()

            imp = impute_EM(data, {}) 

            end <- Sys.time()
            duration <- end - start  

            return(list("imp" = imp, "duration" = duration))
            }}
        """.format(
        args_string
    )

    mmask = np.isnan(X)

    impute_EM = SignatureTranslatedAnonymousPackage(rcode_string, "impute_EM")

    result = impute_EM.imputing(pd.DataFrame(X))

    imp = result.rx2("imp")
    duration = result.rx2("duration")

    # fill to the original matrix
    Ximp = X.copy()
    Ximp[mmask] = imp[mmask]

    return Ximp, duration


def missforest_imputer(X, **kwargs):
    start = time.time()

    imputer = MissForest(
        **kwargs
    )  # n_neighbors=5 # this should be read from cofig file

    Ximp = imputer.fit_transform(Xtrain)
    end = time.time()

    duration = end - start
    return Ximp, duration


def knn_imputer(X, **kwargs):
    start = time.time()

    imputer = KNNImputer()  # n_neighbors=5 # this should be read from cofig file
    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start

    return Ximp, duration


def gain_imputer(X, **kwargs):
    #    start = time.time()
    #
    #    Ximp = gain(X, kwargs)
    #
    #    end = time.time()
    #    duration = end - start
    #
    #    return Ximp, duration
    #
    pass


def ginn_imputer(X, **kwargs):
    pass


def dimv_imputer(X, **kwargs):
    pass
