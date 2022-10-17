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
