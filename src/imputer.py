import os
import sys

root = os.environ.get("ROOT")
sys.path.append(root)

import time
import numpy as np
import pandas as pd

# import to run R code : if got error makee sure : conda install libcblas
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split
from includes.VAE.dataset import generate_dataset
from includes.VAE.train import train

# TODO
# [x] mean
# [x] softimpute (fancyimpute)
# [x] mice (sklearn)
# [x] imputePCA (R)
# [x] missingpy - MissForest: tunning param
# [x] EM (R)
# [x] KNN (sklearn)
# [] ginn
# [] vae
# [x] gain
# [x] dimv
# [] pipeline with Makefile; addding try catch if running multilple algorithm
# [] docker to help install


def mean_imputer(X, **kwargs):
    from sklearn.impute import SimpleImputer

    start = time.time()
    imputer = SimpleImputer(**kwargs)

    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start
    return Ximp, duration


def softimpute_imputer(X, **kwargs):
    from fancyimpute import SoftImpute  # install  then install cvxopt-1.2.6

    start = time.time()

    imputer = SoftImpute(**kwargs)  # default maxit = 1000
    # imputer = SoftImpute()  # default maxit = 1000
    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start

    return Ximp, duration


def mice_imputer(X, **kwargs):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, KNNImputer

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
    print("Input Shape", X.shape)
    mmask = np.isnan(X)
    Ximp = X.copy()
    print("input shape", mmask.shape)

    impute_EM = SignatureTranslatedAnonymousPackage(rcode_string, "impute_EM")

    result = impute_EM.imputing(pd.DataFrame(X))

    imp = result.rx2("imp")
    duration = result.rx2("duration")
    imp = np.asarray(imp)
    print(imp.shape)

    # fill to the original matrix
    Ximp[mmask] = imp[mmask]

    return Ximp, duration[0]


def missforest_imputer(X, **kwargs):
    # this is for the package missingpy
    import sklearn.neighbors._base  # scikit-learn==1.1.2

    sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
    from missingpy import MissForest

    start = time.time()

    imputer = MissForest(
        **kwargs
    )  # n_neighbors=5 # this should be read from cofig file

    Ximp = imputer.fit_transform(X)
    end = time.time()

    duration = end - start
    return Ximp, duration


def knn_imputer(X, **kwargs):
    from sklearn.impute import KNNImputer

    start = time.time()

    imputer = KNNImputer(
        **kwargs
    )  # n_neighbors=5 # this should be read from cofig file
    Ximp = imputer.fit_transform(X)

    end = time.time()
    duration = end - start

    return Ximp, duration


def gain_imputer(X, **kwargs):
    from includes.GAIN.gain import gain

    # return gain(X, self.gain_params);

    start = time.time()

    Ximp = gain(X, kwargs)

    end = time.time()
    duration = end - start

    return Ximp, duration


def vae_imputer(Xmiss, **kwargs):
    # print(Xmiss).shape
    # Xmiss = [sample for sample in Xmiss]
    t0 = time.time()
    X_train, X_valid = train_test_split(Xmiss, test_size=0.2)

    # reshape into image
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_valid = X_valid.reshape((-1, 28, 28, 1))

    train_ds = generate_dataset("MNIST", X_train, None, 32)
    valid_ds = generate_dataset("MNIST", X_valid, None, 32)

    # Fitting data
    model, get_inputs = train(
        run=1,
        method="Zero Imputation",
        train_ds=train_ds,
        valid_ds=valid_ds,
        ds_name="MNIST",
        z_dim=50,
        likelihood="BERNOULLI",
        mixture_components=1,
    )

    print("[+] Inference step")
    Xmiss = Xmiss.reshape((-1, 28, 28, 1))
    Xmiss = generate_dataset("MNIST", Xmiss, None, 32, infer=True)

    results = []
    for example in Xmiss:
        inputs, decoder_b = get_inputs(example)
        (x_logits, scale_logit, pi_logit), q_z, _ = model(inputs, decoder_b)

        x_pred = tf.nn.sigmoid(x_logits)
        x_pred = x_pred.numpy()
        results.append(x_pred)

    results = np.concatenate(results, axis=0)
    t0 = time.time() - t0

    return results, t0


def ginn_imputer(X, **kwargs):
    from inclues.ginn import ginn_run

    start = time.time()
    Ximp = ginn_run(X, y)
    duration = time.time() - start
    return Ximp, duration




def dimv_imputer(X, **kwargs):
    print("X.shape", X.shape)
    n_jobs = kwargs.get("n_jobs")
    train_percent = kwargs.get("train_percent")

    from DIMVImputation import DIMVImputation

    start = time.time()
    imputer = DIMVImputation()
    imputer.fit(X, n_jobs=n_jobs)

    # run cross validation
    best_alpha = imputer.cross_validate(train_percent=train_percent)
    print(
        "Alpha choosen after CV: {} with scores {} ".format(
            imputer.best_alpha, imputer.cv_score
        )
    )

    Ximp = imputer.transform(X, alpha=best_alpha)
    duration = time.time() - start
    return Ximp, duration
