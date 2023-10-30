# Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning (MKOR)

This is the offical implementation for Python simulation of Maximum Knowledge Orthogonality Reconstruction (MKOR), from the following paper: 

  Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning.([WACV2024](accepted, todo: weblink), [arXiv](todo: weblink))  
Feng Wang Senem Velipasalar, and M. Cenk Gursoy  
Department of Electrical Engineering and Computer Science, Syracuse University

---

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/dense_decouple.pdf" width="644" height="501">

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/conv_orthogonal_largeFont.pdf" width="644" height="501">

We propose the FbFTL as an innovative federated learning approach that upload features and outputs instead of gradients to reduce the uplink payload by more than five orders of magnitude. Please refer to the paper for explicit explaination on learning structure, system design, and privacy analysis.


# Results on CIFAR-10 Dataset
In the following table, we provide comparison between federated learning with [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html) (FL), federated transfer learning with FedAvg that updating full model (FTL<sub>f</sub>), federated transfer learning with FedAvg that updating task-specific sub-model(FTL<sub>c</sub>), and FbFTL. All of them learn [VGG16](https://arxiv.org/abs/1409.1556) model on [CIFAR-10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset. For transfer learning approaches, the source models are trained on [ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848?casa_token=QncCRBM1tzAAAAAA:QuoJhjJAHRplmLJ4jcFw5JWdfASjmbIVlvpCrHgTPIFu63gpSUlBeACB78S0AH34qqQnsBOdoQ) dataset. Compared to all other methods, FbFTL reduces the uplink payload by up to five orders of magnitude. 

| | FL | FTL<sub>f</sub> | FTL<sub>c</sub> | FbFTL  |
| ---- | ----- | ---- | ---- | ---- |
| upload batches | 656250 | 193750 | 525000 | 50000 |
| upload parameters per batch | 153144650 | 153144650 | 35665418 | 4096 |
| uplink payload per batch | **4.9 Gb** | **4.9 Gb** | **1.1 Gb** | **131 Kb**  |
| total uplink payload | **3216 Tb** | **949 Tb** | **599 Tb** | **6.6 Gb** |
| total downlink payload | 402 Tb | 253 Tb | 322 Tb | 3.8 Gb |
| test accuracy | 89.1\% | 91.68\% | 85.59\% | 85.59\% |

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/outputs.pdf" width="644" height="501">

# Required packages installation
We use the following packages in the code. The dataset will be automatically downloaded. The computation on a batch size 100 typically takes about 2 minutes on CPU.
torch: 1.13.0+cu117
torchvision: 0.14.0+cu117
torch.version.cuda: 11.7
matplotlib: 3.1.1
os.name: posix
numpy: 1.17.3
torchmetrics: 0.11.0
Device: cpu


# Citation
If you find our work useful in your research, please consider citing:
```
todo
```
