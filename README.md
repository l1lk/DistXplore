# DistXplore-demo
The code demo of DistXplore



## Installation

We have tested DistXplore based on Python 3.6 on Ubuntu 20.04, theoretically it should also work on other operating systems. To get all the dependencies, it is sufficient to run the following command.

`pip install -r requirements.txt`

## The structure of the repository

### DistXplore/dis-guided

This directory contains the core implementation of DistXplore, the subdirectory *profile* and *seeds* provides the subject model and initial seeds. (You can download the pretrained model files from https://drive.google.com/drive/folders/1rgZA2xuMLhcYE40u4llWMxqEsew4rbzb?usp=sharing)

### DistXplore/defence

This directory contains the implementation of three adversarial defense techniques: DISSECTOR, Attack-as-Defense and Data transformation.

### DistXplore/enhancement

This directory contains the code of model retrain and retrain model evaluation.

### Usage

### Distribution-aware testing

We provide a script to generate distribution-aware test samples for LeNet4 model trained on MNIST dataset. You can download other models from the google drive mentioned above.
test
```
test
cd DistXplore/dist-guided
sh generate_demo.sh
```

### Defense

#### Dissector

```
cd DistXplore/defence/dissector
python merge_output_tech.py
       -mode mnist
       -tech bim
       -truth 0
       -target 1
```

The meaning of the options are:

1. **-mode**: the type of the dataset
  
2. **-tech**: the technique to generate the test samples
  
3. **-truth**: for the targeted techniques, set the truth label
  
4. **-target**: for the targeted techniques, set the target label
  

#### Data Transformation

```
cd DistXplore/defence/data transformation
python data_transformation_mnist.py
```

#### Attack-as-Defense

Firstly, to get the attack cost, run

```
cd DistXplore/defence/attack as defence/scripts
python get_attack_cost_mnist.py
```

Then, use the cost recorded to detect the adversarial samples

```
python attack_as_defense_detector.py
       --dataset mnist
       -d knn
       -a JSMA
```

The meaning of the options are:

1. **--dataset**: the type of the dataset
  
2. **-d**: detetor type
  
3. **-a**: attack to use; recommanded to use JSMA, BIM or BIM2.
  
4. **--init**: for the first run, add this argument to train the detector
  

### Enhancement

#### retrain model

```
python mnist_finetune_diversity.py
       -ft_epoch 20
```

**-ft_epoch**: the num of the retrain epoch

#### evaluate model

```
python evaluae.py
```
