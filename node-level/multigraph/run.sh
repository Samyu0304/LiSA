#!/usr/bin/env bash
# Twitch-e
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 0.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 0.01 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 0.1 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99  --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 0.5 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 1.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 2.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 3 --num_layers 2 --device 4 --kld_weight 5.0 --runs 10 --rocauc
# fb-100
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 0.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 0.01 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 0.1 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 0.5 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 1.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 2.0 --runs 10 --rocauc
python mask_main.py --dataset twitch-e --gnn gcn --lr 0.01 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 4 --kld_weight 5.0 --runs 10 --rocauc
