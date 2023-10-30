################### Check the version of required packages ###################
import torch
import torchvision
import matplotlib
import os
import numpy
import torchmetrics
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('matplotlib:', matplotlib.__version__)
print('os.name:', os.name)
print('numpy:', numpy.__version__)
print('torchmetrics:', torchmetrics.__version__)


################### reconstruct CIFAR100 input from VGG16 network with manipulated weights and biases ###################
from VGG16 import VGG16
from LeNet import LeNet
from Modified_LeNet import Modified_LeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import time
import warnings
start_time = time.time()
warnings.filterwarnings("ignore", module="matplotlib\..*")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

# federated learning settings
model_data_name = 'mod_imagenet'  # 'lenet_mnist', 'mod_mnist', 'vgg16_cifar100', 'mod_cifar100', 'vgg16_imagenet', 'mod_imagenet'
unique_batch = True  # True: each sample has different label. False: different samples may have the same label
if unique_batch: user_idx = -1  # index of unique batch (-1 for random) (0 in our experiment)
else: picked_classes = None  # None: random samples, int>0: all samples are from this number of classes
num_image = 1000  # 1000. number of images in experiment, determine num_exp (1000 in our experiment)
if not unique_batch: set_batch_size_rate = 1  # optional if not unique_batch, pick a value > 0.
# batch_size = number_of_classes * batch_size_rate. Best MKOR performance at batch_size_rate<=1
print_size = 100  # None. set int < batch_size to limit number of images to plot and measure

# defense settings
grad_noise = None  # None for no noise, float>0 for std of Gaussian noise on gradient
soteria = None  # None for no prune, 0<int<=100 for percentile of grad to prune with soteria defense

batch_size_rate = 1 if unique_batch else set_batch_size_rate
cmap = 'gray'  # For gray scale image: 'gray' for black/white plot, None for yellow/blue plot
if model_data_name == 'vgg16_cifar100':
    exp_name = '\nCIFAR100 on original VGG16,'
    batch_size = int(100 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.CIFAR100(root='data', train=False, transform=custom_transform, download=True)
    data_size = [3, 224, 224]
    net = VGG16(cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
elif model_data_name == 'lenet_mnist':
    exp_name = '\nMNIST on original LeNet,'
    batch_size = int(10 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='data', train=False, transform=custom_transform, download=True)
    data_size = [1, 28, 28]
    net = LeNet(cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
elif model_data_name == 'mod_cifar100':
    exp_name = '\nCIFAR100 on modified LeNet,'
    batch_size = int(100 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root='data', train=False, transform=custom_transform, download=True)
    data_size = [3, 32, 32]
    net = Modified_LeNet(num_classes=100, data_size=data_size, cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
elif model_data_name == 'mod_mnist':
    exp_name = '\nMNIST on modified LeNet,'
    batch_size = int(10 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='data', train=False, transform=custom_transform, download=True)
    data_size = [1, 28, 28]
    net = Modified_LeNet(num_classes=10, data_size=data_size, cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
elif model_data_name == 'vgg16_imagenet':
    # Manually download ImageNet dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
    exp_name = '\nImageNet on original VGG16,'
    batch_size = int(1000 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageNet(root='data', split="val", transform=custom_transform)
    data_size = [3, 224, 224]
    net = VGG16(num_classes=1000, cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
elif model_data_name == 'mod_imagenet':
    # Manually download ImageNet dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
    exp_name = '\nImageNet on modified LeNet,'
    batch_size = int(1000 * batch_size_rate)
    custom_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageNet(root='data', split="val", transform=custom_transform)
    data_size = [3, 224, 224]
    net = Modified_LeNet(num_classes=1000, data_size=data_size, cmap=cmap, grad_noise=grad_noise, soteria=soteria).to(DEVICE)
else:
    exit('unknown model_data_name' + model_data_name)
num_exp = int(num_image / batch_size)

loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
num_classes = max(dataset.targets) + 1
if unique_batch and batch_size != num_classes:
    raise ValueError('requiring unique_batch, but batch_size != num_classes!', batch_size, num_classes)
if not unique_batch and picked_classes and picked_classes > num_classes:
    raise ValueError('picked_classes > num_classes!', picked_classes, num_classes)
max_SSIMs, avg_SSIMs, max_PSNRs, avg_PSNRs = [], [], [], []

# uncomment if manually set a batch
# imidx_list = []
# print('\nManually set the batch')
# inputs = torch.zeros(batch_size, data_size[0], data_size[1], data_size[2], requires_grad=False)
# labels = torch.zeros(batch_size, requires_grad=False).type(torch.int64)
# for i, imidx in enumerate(imidx_list):
#     inputs[i], labels[i] = dataset[imidx]
# net.process(inputs, labels, DEVICE, batch_size)
# if not print_size:
#     max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
# else:
#     max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
# exit('max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(max_SSIM, avg_SSIM, max_PSNR, avg_PSNR))

for idx_exp in range(num_exp):
    # forward propagation, back propagation, and reconstruction
    if not unique_batch:
        if not picked_classes:  # randomly pick samples
            print('\nRandom batch of all classes, experiment # {:d} / {:d}'.format(idx_exp+1, num_exp))
            for inputs, labels in loader:
                # Client execute: generate gradient update
                net.process(inputs, labels, DEVICE, batch_size)
                # Malicious Server execute: reconstruct input image from gradients
                if not print_size:
                    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
                else:
                    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
                break
        else:  # only choose samples of the first "unique_classes" classes
            print('\nRandom batch of {:.0f} picked classes, experiment # {:d} / {:d}'.format(picked_classes, idx_exp+1, num_exp))
            # generate batch
            inputs = torch.zeros(batch_size, data_size[0], data_size[1], data_size[2], requires_grad=False)
            labels = torch.zeros(batch_size, requires_grad=False).type(torch.int64)
            picked_classes_set = random.sample(range(0, num_classes), picked_classes)
            random_range = list(range(len(dataset)))
            random.shuffle(random_range)
            counter = 0
            for i in random_range:
                _, label = dataset[i]
                if label in picked_classes_set:
                    inputs[counter], labels[counter] = dataset[i]
                    counter += 1
                    if counter >= batch_size:
                        break
            # Client execute: generate gradient update
            net.process(inputs, labels, DEVICE, batch_size)
            # Malicious Server execute: reconstruct input image from gradients
            if not print_size:
                max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
            else:
                max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
    else:  # copy gradient inversion input: 1 sample per class
        print('\nUnique batch of all classes, experiment # {:d} / {:d}'.format(idx_exp+1, num_exp))
        inputs = torch.zeros(batch_size, data_size[0], data_size[1], data_size[2], requires_grad=False)
        labels = torch.zeros(batch_size, requires_grad=False).type(torch.int64) - 1
        if user_idx == -1:
            random_range = list(range(len(dataset)))
            random.shuffle(random_range)
            for j in random_range:
                _, i = dataset[j]
                if labels[i] == -1:
                    inputs[i], labels[i] = dataset[j]
                    if len(torch.where(labels==-1)[0]) == 0:
                        break
        else:
            for i in range(batch_size):
                idx = 0
                for j in range(len(dataset)):
                    _, label = dataset[j]
                    if label == i:
                        if idx == user_idx:
                            inputs[i], labels[i] = dataset[j]
                            break
                        else:
                            idx += 1
        # Client execute: generate gradient update
        net.process(inputs, labels, DEVICE, batch_size)
        # Malicious Server execute: reconstruct input image from gradients
        if not print_size:
            max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
        else:
            max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
        if user_idx != -1:
            user_idx += 1
    max_SSIMs.append(max_SSIM)
    avg_SSIMs.append(avg_SSIM)
    max_PSNRs.append(max_PSNR)
    avg_PSNRs.append(avg_PSNR)


print(exp_name, 'unique batch,' if unique_batch else 'random batch,', 'batch size:', batch_size,
      ', gradient noise {:.4f}'.format(grad_noise) if grad_noise else ', no noise',
      ', soteria {:d}%'.format(soteria) if soteria else ', no soteria')
print('max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(sum(max_SSIMs)/len(max_SSIMs), sum(avg_SSIMs)/len(avg_SSIMs),
                        sum(max_PSNRs)/len(max_PSNRs), sum(avg_PSNRs)/len(avg_PSNRs)))
print('{:.0f} seconds'.format(time.time() - start_time))

# uncomment if try additional experiments on gaussian (differential privacy) defense
input("Press Enter to continue...")
net.grad_noise, net.soteria = 0.1, None
net.process(inputs, labels, DEVICE, batch_size)
if not print_size:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
else:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
print('Gaussian_noise, max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(max_SSIM, avg_SSIM, max_PSNR, avg_PSNR))

# uncomment if try additional experiments on Soteria defense
# input("Press Enter to continue...")
# net.grad_noise, net.soteria = None, 30
# net.process(inputs, labels, DEVICE, batch_size)
# if not print_size:
#     max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
# else:
#     max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
# print('Soteria, max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(max_SSIM, avg_SSIM, max_PSNR, avg_PSNR))

# uncomment if try different network
net = VGG16(num_classes=1000, cmap=cmap, grad_noise=None, soteria=None).to(DEVICE)
input("Press Enter to continue...")
net.process(inputs, labels, DEVICE, batch_size)
if not print_size:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
else:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
print('max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(max_SSIM, avg_SSIM, max_PSNR, avg_PSNR))
input("Press Enter to continue...")
net.grad_noise, net.soteria = 0.1, None
net.process(inputs, labels, DEVICE, batch_size)
if not print_size:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs, labels, dataset.classes)
else:
    max_SSIM, avg_SSIM, max_PSNR, avg_PSNR = net.reconstruction(inputs[:print_size], labels[:print_size], dataset.classes)
print('Gaussian_noise, max ssim, avg ssim, max psnr, avg psnr: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(max_SSIM, avg_SSIM, max_PSNR, avg_PSNR))