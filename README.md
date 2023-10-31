# Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning (MKOR)

This is the offical implementation for Python simulation of Maximum Knowledge Orthogonality Reconstruction (MKOR), from the following paper: 

  Maximum Knowledge Orthogonality Reconstruction with Gradients in Federated Learning.([WACV2024]accepted, [arXiv](https://arxiv.org/abs/2310.19222))  
Feng Wang Senem Velipasalar, and M. Cenk Gursoy  
Department of Electrical Engineering and Computer Science, Syracuse University

---

We propose the MKOR as an innovative as a novel federated learning (FL) input reconstruction method, which can be used with large batches. Our approach maliciously sets the parameters of fully-connected layers to accurately reconstruct the input features, and of convolutional layers to maximize the orthogonality between prior and post knowledge. For both cases, we have proposed an inconspicuous approach to avoid being detected by skeptical clients.

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/dense_decouple.png" width="644">

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/conv_orthogonal_largeFont.png" width="1046">

Experiments have demonstrated that MKOR outperforms other baselines on large batches of the MNIST dataset with LeNet model, and on CIFAR-100 and ImageNet datasets with a VGG16 model by providing better reconstructed images both qualitatively and quantitatively. Our results encourage further research on the protection of data privacy in FL. 

The following is the qualitative comparison between different input reconstruction methods on CIFAR-100 for a batch size of 100. Please refer to the paper for more experiment results, explicit explaination on learning structure, system design, and privacy analysis.

<img src="https://github.com/wfwf10/MKOR/blob/main/diagrams/outputs.png" width="644">

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
