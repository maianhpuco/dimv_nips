import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from src.load_data import load_data, create_split_indices


def train_test_splitting(dataset_name):

    Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = \
            create_split_indices(dataset_name)
    return Xtrain, Xtest, ytrain, ytest, train_indices, test_indices


def grid_search(X, y, n_splits=5):
    print("search", X.shape, y.shape)
    max_depths = [5, 10, 15]
    subsamples = [0.5, 0.7, 1.0]
    learning_rates = [0.001, 0.01, 0.1]

    best_params = {}
    best_accuracy = 0

    num_class = np.unique(y).shape[0]
    print("num class ", num_class)

    kf = KFold(n_splits=n_splits)
    for subsample in subsamples:
        for max_depth in max_depths:
            for learning_rate in learning_rates:

                params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'objective': 'multi:softmax',
                    'num_class': num_class,
                    'eval_metric': 'mlogloss'
                }
                # Initialize score for current hyperparameter combination
                scores = []

                # Train and test model using k-fold cross-validation
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[
                        train_index,
                    ], X[test_index]
                    y_train, y_test = y[
                        train_index,
                    ], y[test_index]

                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)

                    model = xgb.train(params, dtrain)

                    y_pred = model.predict(dtest)
                    scores.append(accuracy_score(y_test, y_pred))

                    # Calculate mean accuracy score across all folds
                accuracy = np.mean(scores)
                print(">> params {}, Avg Acc  {} ".format(params, accuracy))

                # Update best parameters and score if current model performs better
                if accuracy > best_accuracy:
                    best_params = params
                    best_accuracy = accuracy

    print("Done grid search param")
    print("Best hyperparameters:", best_params)
    print("Accuracy score:", best_accuracy)
    return (best_params, best_accuracy)


def classification(Ximp_train, y_train, Xgt_test, y_test, params):

    #X_train, X_test, y_train, y_test = train_test_split(
    #     Ximp, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(Ximp_train, label=y_train)
    dtest = xgb.DMatrix(Xgt_test, label=y_test)

    model = xgb.train(params, dtrain)
    y_pred = model.predict(dtest)

    acc = accuracy_score(y_test, y_pred)

    return acc
