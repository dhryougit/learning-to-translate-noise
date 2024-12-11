<!-- The official pytorch implementation of the paper **[Simple Baselines for Image Restoration (ECCV2022)](https://arxiv.org/abs/2204.04676)** -->

<h1 align="center">Learning to Translate Noise for Robust Image Denoising
</h1>

<p align="center">
<a href="https://arxiv.org/abs/2412.04727"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  &nbsp;
  <a href="https://hij1112.github.io/learning-to-translate-noise/"><img src="https://img.shields.io/badge/Website-ProjectPage-A55D35"></a>
  &nbsp;
</p>

<p align="center">
  <img src="figures/nind.gif" style="width:70%;" />
</p>



---

## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks.

```python
python 3.8.13
pytorch 1.13.0
cuda 11.7
```

```
git clone https://github.com/dhryougit/learning-to-translate-noise.git
cd learning-to-translate-noise
pip install -r requirements.txt
```

We used NVIDIA RTX A6000 D6 48GB for training our models.<br><br>


## QuickStart
Pretraining for image denoising model 
```
python3 -m torch.distributed.launch --nproc_per_node=2 train.py -opt=options/train/nafnet.yml --name=test --launcher pytorch
```
<br>

Training for noise translation network 
```
python3 -m torch.distributed.launch --nproc_per_node=2 train.py -opt=options/train/nafnet_trans.yml --wass_weight=0.05 --spatial_freq_weight=0.002 --seed=0 --noise_injection_level=100 --name=NTNet --launcher pytorch
```
<br>

For test
```
python3 test.py -opt=options/test/nafnet_trans.yml
```
<br>

## Dataset

Training dataset : [SIDD](https://abdokamel.github.io/sidd/#sidd-medium), CBSD400, WED

Evaluation datasets : [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset), [CC](https://github.com/csjunxu/MCWNNM-ICCV2017), HighISO, iPhone, Huawei, OPPO, Sony, Xiaomi.

Additional real-world noise datasets can be downloaded from "https://github.com/ZhaomingKong/Denoising-Comparison"<br>



## Results and Pre-trained model


| Metric | SIDD   | Poly   | CC     | HighISO | iPhone | Huawei | OPPO   | Sony   | Xiaomi | OOD Avg. |
|--------|--------|--------|--------|---------|--------|--------|--------|--------|--------|----------|
| PSNR   | 39.17  | **38.67**  | **37.82**  | **39.94**  | **41.94**  | **39.74**  | **40.45**  | **44.17**  | **36.14**  | **39.86**  |
| SSIM   | 0.9566 | **0.9851** | **0.9876** | **0.9853** | **0.9805** | **0.9778** | **0.9796** | **0.9869** | **0.9745** | **0.9822** |


Our pretrained NAFNet and noise translation network can be downloaded from (https://drive.google.com/drive/folders/1Wy7lSRM7yrceQs5DGUJFBo8Mh9kQNExj?usp=sharing)
