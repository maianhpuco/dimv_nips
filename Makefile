
dev:
	ROOT=$${PWD} python exp/mono/imputing.py --ds mnist --algo vae --dryrun 1


ALGOS = mean softimpute mice imputepca em knn gain dimv # ginn, vae 

# include root folder into python search path
ROOT=$${PWD}

test:
	echo ${ROOT}

# ifneq ($algo, )
# 	ALGOS = $algo
# endif 

DATA = mnist
DRYRUN = 0 


mono_missing:
	for one_ds in $(DATAS); do\
		ROOT=$${PWD} python3 exp/mono/missing.py --ds $$one_ds ;\
	done 

rand_missing:
	for one_ds in $(DATA); do\
		ROOT=$${PWD} python3 exp/rand/missing.py --ds $$one_ds;\
	done 

# --------------------------------------------------
# Run missing crreation, imputation
# --------------------------------------------------
# Example:
# make DATA=mnist mono_missing 

# RUN ALL ALGOS: 
# make DATA=mnist mono_imputing
#
# CUSTOM INPUT: 
# make DATA=mnist ALGOS="mean softimpute"  mono_imputing
#
# CUSTOM INPUT for DRYRUN 
# make DATA=mnist ALGOS="mean softimpute dimv" DRYRUN=1 mono_imputing

mono_imputing:
	for algo in $(ALGOS); do \
		python3 exp/mono/imputing.py --ds $(DATA) --algo $$algo --dryrun $(DRYRUN);\
	done 

mono_classifying_no_grid_search:
	for algo in $(ALGOS); do \
		python3 exp/mono/classifying.py --ds $(DATA) --algo $$algo --dryrun $(DRYRUN);\
	done

mono_classifying: 
	python3 exp/mono/grid_search.py --ds $(DATA) --dryrun $(DRYRUN);\
	for algo in $(ALGOS); do \
		python3 exp/mono/classifying.py --ds $(DATA) --algo $$algo --dryrun $(DRYRUN);\
	done

rand_imputing_one_algo:
	for exp in {1..9}; do\
		for data in $(DATAS); do \
			ROOT=$${PWD} python3 exp/rand/imputing.py --ds $$data --algo $$ALGO --dryrun $(DRYRUN);\
		done \
	done


rand_imputing:
	for algo in $(ALGOS); do \
		ROOT=$${PWD} python3 exp/rand/imputing.py --ds $(DATA) --algo $$algo --dryrun $(DRYRUN);\
	done 



rand_imputing_loop:
	for exp in {1..9}; do\
		for data in $(DATAS); do\
			for algo in $(ALGOS); do \
				ROOT=$${PWD} python3 exp/rand/imputing.py --ds $$data --exp_num $$exp --algo $$algo --dryrun $(DRYRUN);\
			done \
		done 
#  Download raw dataset
# --------------------------------------------------
#  To download mnist + fashion_mnist : make download_all
download_all: init_ds download_mnist download_fashion_mnist

init_ds:
	mkdir -p data
	mkdir -p data/raw
	mkdir -p data/raw/mnist
	mkdir -p data/raw/fashion_mnist

download_mnist:
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz	-o data/raw/mnist/train-images-idx3-ubyte.gz  
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz	-o data/raw/mnist/train-labels-idx1-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz		-o data/raw/mnist/t10k-images-idx3-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz		-o data/raw/mnist/t10k-labels-idx1-ubyte.gz

download_fashion_mnist:
	curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz 	-o data/raw/fashion_mnist/train-images-idx3-ubyte.gz
	curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz 	-o data/raw/fashion_mnist/train-labels-idx1-ubyte.gz
	curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz 	-o data/raw/fashion_mnist/t10k-images-idx3-ubyte.gz 
	curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz 	-o data/raw/fashion_mnist/t10k-labels-idx1-ubyte.gz


