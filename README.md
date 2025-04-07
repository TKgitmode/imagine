# Towards Scale-Aware Low-Light Enhancement via Structure-Guided Transformer Design

This repo is our solution for [NTIRE 2025 Low Light Image Enhancement Challenge](https://codalab.lisn.upsaclay.fr/competitions/21636).

## Description

![net](https://github.com/minyan8/imagine/blob/main/figure/net.png)

## Installation, Training, and Testing

### Create Environment

1. Create Conda Environment
```
conda create --name imagine python=3.10
conda activate imagine
```

2. Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
pip install numpy==1.26.1
```

3. Install BasicSR
```
python setup.py develop --no_cuda_ext
```


### Pre-trained Model
- [Pre-trained Model for NTIRE 2025 Low Light Image Enhancement Challenge](https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/Em4rtdZsS3NKtE2K-pTXCXsBSrwmB_gPwXtd0eldBUn6Ig?e=pAZVvC).

### Our Submission on Test Sever
- [Our Test Output](https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/EpN54Q4bzO9DteK4tntg_eYB4X8XzlqI8A6HNXtAUEALSw?e=7zbUrM).

### Testing
Download above saved models and put it into the folder ./Enhancement/weights. To test the model, you need to specify the input image path (`args.input_dir`), the input structure prior path(`args.input_dir_s`), and pre-trained model path (`args.weights`) in `./Enhancement/test.py`. Then run
```bash
cd Enhancement
python test.py 
```
You can check the output in `test-results-ntire25`.


### Contact
If you have any question, please feel free to contact us via dongw22@mcmaster.ca.
