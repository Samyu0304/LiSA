#!/bin/bash
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset SPMotif-0.33 --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset SPMotif-0.5 --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset SPMotif-0.7 --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset SPMotif-0.9 --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset DD --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset MUTAG --layers 2 --batch_size 256  --hiddens 64 --K 3
python  ../multigib_main.py --cuda 4 --kld_weight 0.1 --dist_weight 0.1 --lr_a 0.001  --lr 0.001 --dataset mnist --layers 2 --batch_size 256  --hiddens 64 --K 3