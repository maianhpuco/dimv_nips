import os
import sys 
sys.path.append("") #run from the root directory 
 
from src.load_data import load_data_mnist
from src.utils import create_image_monotone_missing, get_directory
import argparse 


import yaml

with open('exp/cfg.yml', 'r') as f:
    cfg_loaded = yaml.safe_load(f) 
    print(cfg_loaded)


def create_missing(dataset_name, missing_rates):      
    Xtrain, ytrain, Xtest, ytest = load_dataset(dataset_name)
    
    w = cfg_loaded['data_meta'][dataset_name]['im_width']
    h = cfg_loaded['data_meta'][dataset_name]['im_height']
 
    missing_directory = get_directory(
            stage = 'missing', 
            mono_or_rand = 'mono', 
            dataset_name = dataset_name
            )

    for mrate in missing_rates:

        X_train_missing, _ = create_image_monotone_missing(
                X_train_missing, 
                .5,
                mrate, mrate,
                w, h 
                )
        train_name = np.savez_compressed(
                missing_directory + 'Xtrain.npz', X_train_missing)

        X_test_missing, _ = create_image_monotone_missing(
                X_test_missing, 
                .5,
                mrate, mrate,
                w, h 
                )
        test_name = np.savez_compressed(
                missing_directory + 'Xtest.npz', X_train_missing)

         

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--missing_rates", type=List, default = [.6, .5, .4])

    args = parser.parse_args() 

    create_missing(
            args.datasets_name, 
            args.missing_rates
            )
