#!/usr/bin/env bash
# Twitch-e
#two bestpython ours_main.py --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 7 --rocauc
#two bestpython ours_main.py --method eerm --gnn gcn --lr 0.001 --weight_decay 1e-3 --num_layers 2 --device 7 --rocauc
python mask_main.py --gnn sage --device 2 --dist_weight 0.0 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --dist_weight 0.01 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --dist_weight 0.05 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --dist_weight 0.1 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --dist_weight 0.5 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --dist_weight 1.0 --lr_a 0.01

python mask_main.py --gnn sage --device 2 --kld_weight 0.0 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --kld_weight 0.01 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --kld_weight 0.05 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --kld_weight 0.1 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --kld_weight 0.5 --lr_a 0.01
python mask_main.py --gnn sage --device 2 --kld_weight 1.0 --lr_a 0.01
# fb-100
#python ours_main.py --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2  --dataset fb1001 --device 7 --epochs 200

