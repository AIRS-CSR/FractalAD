# FractalAD: A simple industrial anomaly detection method using fractal anomaly generation and backbone knowledge distillation

This is the official implementation for FractalAD (Pytorch)

##Abstract

Although industrial anomaly detection (AD) technology has made significant progress in recent years, generating realistic anomalies and learning priors of normal remain challenging tasks. In this study, we propose an end-to-end industrial anomaly detection method called FractalAD. Training samples are obtained by synthesizing fractal images and patches from normal samples. This fractal anomaly generation method is designed to sample the full morphology of anomalies. Moreover, we designed a backbone knowledge distillation structure to extract prior knowledge contained in normal samples. The differences between a teacher and a student model are converted into anomaly attention using a cosine similarity attention module. The proposed method enables an end-to-end semantic segmentation network to be used for anomaly detection without adding any trainable parameters to the backbone and segmentation head, and has obvious advantages over other methods in training and inference speed.. The results of ablation studies confirmed the effectiveness of fractal anomaly generation and backbone knowledge distillation. The results of performance experiments showed that FractalAD achieved competitive results on the MVTec AD dataset and MVTec 3D-AD dataset compared with other state-of-the-art anomaly detection methods.

Paper URL is here[https://arxiv.org/abs/2301.12739]

## Dataset Preparation

    ├── fractals
    ├── fag.py
    ├── mvtec.py
    ├── networks.py
    ├── run.py
    ├── tools.py
    ├── MVTec_AD
        ├── bottle
        ├── cable
        └── ...

## Train

    python run.py

## Test

    python run.py --phase test
