bazel-bin/lm_1b/lm_1b_eval --mode sample --pbtxt data/graph-2016-09-10.pbtxt --vocab_file data/vocab-2016-09-10.txt --prefix "" --ckpt 'data/ckpt-*' --num_samples 1000 > big_lm_examples.txt
