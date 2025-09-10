# DBAF

Code for Paper "Disentangle Before Anonymize: A Two-stage Framework for Attribute-preserved and Occlusion-robust De-identification".

![network](/imgs/overview.png)

## Setup

- **Get code**

```
git clone https://github.com/mrzhu-cool/DBAF.git
```

- **Build environment**

```
cd DBAF
# use anaconda to build environment 
conda create -n DBAF python=3.10
conda activate DBAF
# install torch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install packages
pip install -r requirements.txt
```

- **Download models**
  - Download the pretrained models [Google](https://drive.google.com/drive/folders/1hrp3rLlODcYe3h_uNQjb5Knm_rAC5OOO?usp=sharing) and place them in the `pretrain` folder.

- **The final folder should be like this:**

```
DBAF
  └- datasets
    └- train
    └- test
  └- pretrain
    └- parsenet.pth
    └- ...
  └- train_anonymization_stage.py
  └- train_disentanglement_stage.py
  └- test.py
```

## Quick Start

- **Train Disentanglement(Stage 1)**

```
python train_disentanglement_stage.py --train_img_dir <train_dir>  --test_img_dir <test_dir>  --save_dir <save_dir>  --device <device>
```

- **Train Anonymization(Stage 2)**

```
python train_anonymization_stage.py --train_img_dir <train_dir>  --test_img_dir <test_dir>  --save_dir <save_dir>  --device <device>
```

- **Test**

```
python test.py --test_img_dir <test_dir>  --save_dir <save_dir>  --device <device>
```

### Acknowledgments

  * This code builds heavily on **[stylegan2](https://github.com/NVlabs/stylegan2)** and [e4e](https://github.com/omertov/encoder4editing). Thanks for open-sourcing!
