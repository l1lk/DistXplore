time python image_fuzzer.py -i ../test_seeds/mmd_ga_seed_svhn -o ./deephunter_outputs/svhn_resnet_ga_kmnc_iter_5000_efficient/outputs_50 -model svhn_resnet -criteria kmnc -max_iteration 5000 -random 0 -select prob -gpu_index 0 --save_path ./tmp/svhn_vgg