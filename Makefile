ALL_ALGOS := mean, 

.PHONY: train
train:
    @for arg in $(wordlist 2,4,$(ARGS)); do \
        python train.py $(word 1,$(ARGS)) $$arg; \
    done 
