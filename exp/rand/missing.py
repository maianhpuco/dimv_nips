import os
import sys 
sys.path.append("") #run from the root directory 
 
from src.load_data import load_data
from src.utils import create_randomly_missing, get_directory
import argparse 
import numpy as np

import yaml

with open('exp/cfg.yml', 'r') as f:
    cfg_loaded = yaml.safe_load(f) 


def create_missing(dataset_name, missing_rates):      
    X, y = load_data(dataset_name)
    
    for mrate in missing_rates:

        missing_directory = get_directory(
            stage = 'missing', 
            mono_or_rand = 'rand', 
            dataset_name = dataset_name, 
            mrate = mrate 
            )


        Xmiss = create_randomly_missing(X, mrate) # create_randomly_missing return only 1 value
        file_name = np.savez_compressed(
                missing_directory + 'Xmiss.npz', Xmiss) # typo
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create missing data (randomly)')
    parser.add_argument("--ds", type=str, default=None)
    # parser.add_argument("--missing_rates", type=float, default = [.1, .2, .3, .4, .5, .6, .7, 8, .9])

    missing_rates = [i / 10. for i in range(1, 10)]
    print(missing_rates);

    args = parser.parse_args() 
    create_missing( args.ds , missing_rates )
