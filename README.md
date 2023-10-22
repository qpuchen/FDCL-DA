## Graphical Abstract
<div style="text-align:justify"> Ultra-fine-grained visual classification (ultra-FGVC) targets at classifying sub-grained categories of fine-grained objects. This inevitably poses a challenge, i.e., classifying highly similar objects with limited samples. Our method ingeniously marries the powerful Swin-Transformer framework with two self-supervised approaches, crafting a strategy specifically designed to grapple with the unique challenges posed by ultra-FGVC. Through an exploration of  foreground-background feature distillation (FBFD) and contrastive feature learning (CFL) modules and their impact on data representation and feature extraction, this paper unfolds a promising avenue to bolster performance in ultra-FGVC tasks. A given image is first projected into two different views via the following operations: 1) standard augmentation and 2) auxiliary augmentation. Then the two views are sent to the backbone network for feature extraction. The core design is two self-supervised modules, namely the FBFD module and the CFL module. Two self-supervised modules are applied to standard augmentation data. By incorporating these two self-supervised modules, the network acquires more knowledge from the intrinsic structure of the input data, which improves the generalization ability of limited training samples. </div>


<img src='figs/method.jpg' width='1280' height='350'>

## 1. Environmental settings
+ CUDA==11.4
+ Python 3.9.12
+ pytorch==1.12.1
+ torchvision==0.12.0+cu113
+ tensorboard
+ scipy
+ ml_collections
+ tqdm
+ pandas
+ matplotlib
+ imageio
+ timm
+ yacs
+ scikit-learn
+ opencv-python


## 2. Download the pre-trained Swin Transformer models

```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```
## 3. Download Datasets
You can download the datasets from the links below:

+ [Cotton80, SoyLocal, SoyGlobal, SoyGene, and SoyAgeing](https://maxwell.ict.griffith.edu.au/cvipl/UFG_dataset.html).


## 4. Loading data
If the result is the directory below, use **ImageDataset()** in `./data/build.py `to load the dataset

    ./datasets/soybean_gene/
    ├── classA
    ├── classA
    ├── ...
    └── classN

If the result is the directory below, use **Cultivar()** in `./data/build.py` to load the dataset

    ./datasets/soybean_gene/
    └── anno
       ├── train.txt
       └── test.txt
    └──images
       ├── ImageA
       ├── ImageB
       ├── ...
       └── ImageN

## 5. Training
Using the scripts on scripts directory to train the model, e.g., train on SoybeanGene dataset.

```
sh ./scripts/run_gene.sh
```

## 6. Evaluation
Using the scripts on scripts directory to evaluate the model, e.g., evaluate on SoybeanGene dataset.

```
sh ./scripts/test_gene.sh
```