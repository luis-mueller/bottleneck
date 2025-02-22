#!/bin/bash

for seed in 7 31 91 114 61 12 832 24 138 128
do
    python main.py --task NEIGHBORS_MATCH --eval_every 1000 --depth 2 --num_layers 3 --batch_size 64 --type Transformer --seed $seed --attention_dropout 0.0 --no_residual --no_activation --no_layer_norm
    python main.py --task NEIGHBORS_MATCH --eval_every 100 --depth 3 --num_layers 4 --batch_size 64 --type Transformer --seed $seed --attention_dropout 0.0 --no_residual --no_activation --no_layer_norm
    python main.py --task NEIGHBORS_MATCH --eval_every 100 --depth 4 --num_layers 5 --batch_size 1024 --type Transformer --seed $seed --attention_dropout 0.0 --no_residual --no_activation --no_layer_norm
    python main.py --task NEIGHBORS_MATCH --eval_every 100 --depth 5 --num_layers 6 --batch_size 1024 --type Transformer --seed $seed --attention_dropout 0.0 --no_residual --no_activation --no_layer_norm
    python main.py --task NEIGHBORS_MATCH --eval_every 100 --depth 6 --num_layers 7 --batch_size 1024 --type Transformer --seed $seed --attention_dropout 0.0 --no_residual --no_activation --no_layer_norm
done