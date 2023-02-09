#!/usr/bin/env bash

python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=0
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=1
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=2
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=3
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=4
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=5
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=6
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=7
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=8
python dist_gen_diff_cifar.py occl 1 .5 5 1 100 30 .25 --target_model=2 -seed_idx=9
