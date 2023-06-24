#!/usr/bin/env bash

# ogb-arxiv
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.1 --inner_steps 10
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.05 --inner_steps 10
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.01 --inner_steps 10
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.005 --inner_steps 10
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.001 --inner_steps 10
#this is the best
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.0005 --inner_steps 10
#python main.py --gnn appnp --lr 0.01 --lr_a 0.01 --dropout 0.5 --K 3 --device 1 --dist_weight 0.0001 --inner_steps 10

python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.1 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.05 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.01 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.005 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.001 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.0005 --inner_steps 10
python main.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.0 --K 3 --device 0 --dist_weight 0.0001 --inner_steps 10