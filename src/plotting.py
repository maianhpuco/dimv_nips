import os
import sys
sys.path.append("") 

from src.utils import get_path_int_file  



def plotting(dataset_name, missing_rates, exp_nums):
    for exp_num in exp_nums:
        for mrate in missing_rates:
            #get nan row
            missing_path = get_save_path(dataset_name, mrate, exp_num, "missing", "mono")
            missing_f = open(os.path.join(missing_path, 'Xmiss.npz'), 'r') 
            Xmiss = np.load(missing_f)  
            nan_rows = np.where(np.sum(np.isnan(Xmiss), axis = 1 ) > 0)[0]
            
            #read imp data and plot 
            imp_path =  get_save_path(dataset_name, mrate, exp_num, "exp") 
            files = os.listdir(dir_path)
            imp_pattern = r"X_imp_(.*)\.npz"
            
            missing_path = get_save_path(dataset_name, mrate, exp_num)  
            for fnam in files:
                if re.match(imp_pattern, fname):
                    f = open(os.path.join(dir_path, fname)) 
                    imp = f.load(f                




if __name__ == "__main__" 
