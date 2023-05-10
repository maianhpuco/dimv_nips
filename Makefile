ALGOS = mean softimpute 

ifneq ($algo, )
	ALGOS = $algo
endif 

DATA = mnist
DRYRUN = False 
mono_all:

mono_missing:
	python3 exp/mono/missing.py --ds $(DATA)

# --------------------------------------------------
# Run missing crreation, imputation
# --------------------------------------------------
# Example:
# make DATA=mnist mono_missing 
#
# RUN ALL ALGOS: 
# make DATA=mnist mono_imputing
#
# CUSTOM INPUT: 
# make DATA=mnist ALGOS="mean softimpute"  mono_imputing
#
# CUSTOM INPUT: 
# make DATA=mnist ALGOS="mean softimpute" DRYRUN=TRUE mono_imputing

mono_imputing:
	for algo in $(ALGOS); do \
		python3 exp/mono/imputing.py --ds $(DATA) --algo $$algo --dryrun $DRYRUN;\
	done 
	






# ALL_ALGOS := mean, 
# 
# .PHONY: train
# train:
#     @for arg in $(wordlist 2,4,$(ARGS)); do \
#         python train.py $(word 1,$(ARGS)) $$arg; \
#     done 
# --------------------------------------------------
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


