#!/usr/bin/env bash

# elliptic
python main.py --dist_weight 1.0  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.0
python main.py --dist_weight 0.5  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.0
python main.py --dist_weight 0.1  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.0
python main.py --dist_weight 0.05  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.0
python main.py --dist_weight 0.01  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.0
python main.py --dist_weight 1.0  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5
python main.py --dist_weight 0.5  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5
python main.py --dist_weight 0.1  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5
python main.py --dist_weight 0.05  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5
python main.py --dist_weight 0.01  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5