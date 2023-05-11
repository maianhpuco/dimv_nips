import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.load_data import load_data 

def grid_search(X, y):


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
   
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
   
    max_depths = [5, 7, 10]
    learning_rates = [0.01, 0.05, 0.1]
   
    best_params = {}
    best_accuracy = 0
   
    for max_depth in max_depths:
        for learning_rate in learning_rates:
            params = {
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'objective': 'multi:softmax',
                'num_class': np.unique(y).shape[0],
                'eval_metric': 'mlogloss'
             }
            model = xgb.train(params, dtrain)
   
            y_pred = model.predict(dtest)
   
            accuracy = accuracy_score(y_test, y_pred)
   
            if accuracy > best_accuracy:
                best_params = params
                best_accuracy = accuracy
    
    print("Done grid search param")
    print("Best hyperparameters:", best_params)
    print("Accuracy score:", best_accuracy)
    return best_params 


def classification(Ximp, y, params):

    X_train, X_test, y_train, y_test = train_test_split(
         Ximp, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test) 

    
    model = xgb.train(params, dtrain)
   
    y_pred = model.predict(dtest)
   
    acc = accuracy_score(y_test, y_pred)

    return acc 


    
    
