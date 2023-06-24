#!/usr/bin/env bash
# fb100
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.1 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 3 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.1 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 4 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.1 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 5 --num_layers 2 --device 6  --runs 5

python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.1 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 3 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.1 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 4 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.1 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 5 --num_layers 2 --device 6  --runs 5

python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.01 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 3 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.01 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 4 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.01 --display_step 199 --kld_weight 0.01 --dist_weight 0.1 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 5 --num_layers 2 --device 6  --runs 5

python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.01 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 3 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.01 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 4 --num_layers 2 --device 6  --runs 5
python mask_main.py --dataset fb1003 --gnn gcn --lr 0.001 --display_step 199 --kld_weight 0.01 --dist_weight 0.01 --early_stopping 20 --warm_up 30 --epochs 200 --weight_decay 1e-3 --K 5 --num_layers 2 --device 6  --runs 5