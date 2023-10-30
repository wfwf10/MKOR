import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math

class VGG16(nn.Module):
    # input size: [-1, 3, 224, 224]
    def __init__(self, channel=3, hidden=25088, num_classes=100, evaluate=True, param_noise=None, grad_noise=None, soteria=None, cmap='gray'):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv1[0], mode_mod='same', channel_in_mod=3, add_negative=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.manipulate_param_conv(conv_layer=self.conv2[0], mode_mod='same', channel_in_mod=6, add_negative=False)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv3[0], mode_mod='same', channel_in_mod=6, add_negative=False)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.manipulate_param_conv(conv_layer=self.conv4[0], mode_mod='same', channel_in_mod=6, add_negative=False)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv5[0], mode_mod='same', channel_in_mod=6, add_negative=False)

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv6[0], mode_mod='same', channel_in_mod=6, add_negative=False)

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.manipulate_param_conv(conv_layer=self.conv7[0], mode_mod='4_dir', channel_in_mod=6, add_negative=False)

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv8[0], mode_mod='same', channel_in_mod=24, add_negative=False)

        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv9[0], mode_mod='same', channel_in_mod=24, add_negative=False)

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.manipulate_param_conv(conv_layer=self.conv10[0], mode_mod='4_dir', channel_in_mod=24, add_negative=False)

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv11[0], mode_mod='same', channel_in_mod=96, add_negative=False)

        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU())
        self.manipulate_param_conv(conv_layer=self.conv12[0], mode_mod='same', channel_in_mod=96, add_negative=False)

        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.manipulate_param_conv(conv_layer=self.conv13[0], mode_mod='4_dir', channel_in_mod=96, add_negative=False)

        self.dense1 = nn.Sequential(
            nn.Linear(hidden, 4096),
            nn.ReLU()
        )
        self.manipulate_param_dense(dense_layer=self.dense1[0], mode_mod='first', add_negative=False)

        self.dense2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.manipulate_param_dense(dense_layer=self.dense2[0], mode_mod='ones', add_negative=False)

        self.dense3 = nn.Sequential(
            nn.Linear(4096, self.num_classes)
        )
        self.manipulate_param_dense(dense_layer=self.dense3[0], mode_mod='ones', add_negative=False)

        if param_noise:
            self.param_add_Gaussian_noise(param_noise)
        self.grad_noise = grad_noise
        self.soteria = soteria

        self.evaluate = evaluate
        if self.evaluate:
            from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            self.mse = MeanSquaredError()
            self.psnr = PeakSignalNoiseRatio()
        self.cmap = cmap

        self.smooth_filter = self.generate_smooth_filter()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        feature = out.reshape(out.size(0), -1)
        out = self.dense1(feature)
        out = self.dense2(out)
        out = self.dense3(out)
        return out, feature


    def param_add_Gaussian_noise(self, param_noise_std):
        with torch.no_grad():
            for conv in [self.conv1[0], self.conv2[0], self.conv3[0], self.conv4[0], self.conv5[0], self.conv6[0], self.conv7[0],
                         self.conv8[0], self.conv9[0], self.conv10[0], self.conv11[0], self.conv12[0], self.conv13[0]]:
                conv_std = 1. / math.sqrt(conv.weight.size(1) * conv.weight.size(2) * conv.weight.size(3))
                conv.weight.add_(torch.randn(conv.weight.size()) * conv_std * param_noise_std)
                conv.bias.add_(torch.randn(conv.bias.size()) * conv_std * param_noise_std)
            for dense in [self.dense1[0], self.dense2[0], self.dense3[0]]:
                dense_std = 1. / math.sqrt(dense.weight.size(1))
                dense.weight.add_(torch.randn(dense.weight.size()) * dense_std * param_noise_std)
                dense.bias.add_(torch.randn(dense.bias.size()) * dense_std * param_noise_std)


    def grad_add_gaussian_noise(self):
        with torch.no_grad():
            for name,param in self.named_parameters():
                if name=='smooth_filter.weight':
                    continue
                std = torch.std(param.grad) # 1. / math.sqrt(torch.numel(param.grad))
                param.grad.add_(torch.randn(param.size()) * std * self.grad_noise)


    def process(self, inputs, labels, DEVICE, batch_size=100):
        if self.soteria:
            import numpy as np
            inputs.requires_grad = True
        minibatch_size = 100
        self.zero_grad()
        if batch_size <= minibatch_size:
            outs, features = self.forward(inputs)
            if self.soteria:
                deviation_f1_target = torch.zeros_like(features)
                deviation_f1_x_norm = torch.zeros_like(features)
                for f in range(deviation_f1_x_norm.size(1)):
                    deviation_f1_target[:, f] = 1
                    features.backward(deviation_f1_target, retain_graph=True)
                    deviation_f1_x = inputs.grad.data
                    deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (features.data[:, f]+1e-10)
                    self.zero_grad()
                    inputs.grad.data.zero_()
                    deviation_f1_target[:, f] = 0
                # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
                deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), self.soteria)
                mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
            criterion = nn.NLLLoss().to(DEVICE)
            loss = criterion(outs, labels)
            loss.backward()
        else:
            if self.soteria:
                exit('Not sufficient memory to run soteria defense, consider increase minibatch_size.')
            print('Start Generating gradients')
            for minibatch in range(0, batch_size, minibatch_size):
                print('minibatch from image', minibatch, 'to', minibatch+minibatch_size)
                outs, _ = self.forward(inputs[minibatch : minibatch+minibatch_size])
                criterion = nn.NLLLoss().to(DEVICE)
                loss = criterion(outs, labels[minibatch : minibatch+minibatch_size])
                loss.backward()
            print('Start Reconstruction')
        if self.grad_noise:
            self.grad_add_gaussian_noise()
        if self.soteria:
            self.dense1[0].weight.grad = self.dense1[0].weight.grad * torch.Tensor(mask).to(DEVICE)


    def manipulate_param_conv(self, conv_layer, mode_mod, channel_in_mod, add_negative=False):
        # manipulate weight and bias in conv2d layer with kernel_size=3, padding=1
        # weight shape: out_ch; in_ch; kernel[0]; kernel[1]
        with torch.no_grad():
            conv_layer.bias = torch.nn.Parameter(torch.zeros_like(conv_layer.bias))
            center = torch.zeros(3, 3)
            center[1, 1] = 1
            if mode_mod == 'same':  # keep the same in the first channel_in_mod channels (or channel_in_mod*2 for add_negative=True)
                if add_negative:  # generate negative paths in the first conv layer
                    conv_layer.weight[:channel_in_mod * 2] = torch.zeros(channel_in_mod * 2, conv_layer.weight.shape[1], 3, 3)
                    for i in range(channel_in_mod):
                        conv_layer.weight[i, i] = center
                        conv_layer.weight[i + channel_in_mod, i] = - center
                        conv_layer.bias[i + channel_in_mod] = torch.ones(1)
                else:
                    conv_layer.weight[:channel_in_mod] = torch.zeros(channel_in_mod, conv_layer.weight.shape[1], 3, 3)
                    for i in range(channel_in_mod):
                        conv_layer.weight[i, i] = center
            elif mode_mod == '4_dir':
                right = torch.zeros(3, 3)
                right[1, 2] = 1
                lower = torch.zeros(3, 3)
                lower[2, 1] = 1
                lower_right = torch.zeros(3, 3)
                lower_right[2, 2] = 1
                conv_layer.weight[:channel_in_mod * 4] = torch.zeros(channel_in_mod * 4, conv_layer.weight.shape[1], 3, 3)
                for i in range(channel_in_mod):
                    conv_layer.weight[i, i] = center
                    conv_layer.weight[i + channel_in_mod, i] = right
                    conv_layer.weight[i + channel_in_mod * 2, i] = lower
                    conv_layer.weight[i + channel_in_mod * 3, i] = lower_right


    def manipulate_param_dense(self, dense_layer, mode_mod, add_negative=False):
        # mode_mod: 'first', 'ones'
        if add_negative:
            exit('TODO: Implement two paths for each class, only when activation function has negative outputs such as Leaky ReLU.')
        with torch.no_grad():
            # channel_in_mod = self.num_classes if add_negative==False else self.num_classes * 2
            # print('dense_layer.weight.shape', dense_layer.weight.shape)  # [4096, 25088]
            # print('dense_layer.bias.shape', dense_layer.bias.shape)  # [4096]
            dim_out, dim_in = dense_layer.weight.shape[0], dense_layer.weight.shape[1]
            if mode_mod == 'first':  # the first manipulated layer, fully connected
                # positive maximum weights (instead of uniform distribution), to avoid quantization error
                dense_layer.weight = nn.Parameter(torch.ones_like(dense_layer.weight) / dim_in)
                dense_layer.bias = nn.Parameter(torch.zeros_like(dense_layer.bias))
            elif mode_mod == 'ones':  # the layers passing features from different classes separately
                # identity weight matrices
                dense_layer.weight = nn.Parameter(torch.eye(dim_out, dim_in))
                dense_layer.bias = nn.Parameter(torch.zeros(dim_out))
            else:
                raise ValueError('mode_mod not found:', mode_mod)


    def generate_smooth_filter(self, sigma=10):
        with torch.no_grad():
            kernel_size = 5
            smth_filter = nn.Conv2d(3, 3, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), padding_mode='replicate', \
                                    bias=False)
            smth_filter.weight = torch.nn.Parameter(torch.zeros_like(smth_filter.weight))
            # Create a Gaussian filter: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
            mean = (kernel_size - 1) / 2.
            variance = sigma ** 2.
            gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
            for i in range(3):
                smth_filter.weight[i, i] = gaussian_kernel
            smth_filter.requires_grad = False
            return smth_filter


    def reconstruct_single_class(self, c, dense_grads, q_level=False):
        ### reconstruct input corresponding to selected class c. Set quantization level q_level to an int if necessary
        with torch.no_grad():
            ### reconstruct dense input of class c (conv output)
            out = torch.flatten(dense_grads[0][c, :]) / dense_grads[1][c]

            ### reconstruct overall input
            # reconstruct an input sample from one output features with shape [25088], sequence: '4_dir', '4_dir', '4_dir'
            if q_level:  # quantization
                out = (out*q_level).type(torch.int64).type(torch.float32)/q_level
            out = out.reshape(512, 7, 7)
            # find local maximum and minimum
            recon_max = torch.ones(3, 224, 224, requires_grad=False)
            recon_min = torch.zeros(3, 224, 224, requires_grad=False)
            # iterate over first 384 channels, get max and min value
            for ch_1 in range(4):  # first 4_dir + maxpool layer
                for ch_2 in range(4):  # second 4_dir + maxpool layer
                    for ch_3 in range(4):  # third 4_dir + maxpool layer
                        for color in range(3):  # 3 color channels
                            ch = (ch_1 + ch_2*4 + ch_3*16) * 6 + color  # output channel index
                            h_bias = ((ch_1//2) + (ch_2//2)*2 + (ch_3//2)*4) * 4  # height bias from 4_dir layers
                            w_bias = ((ch_1%2) + (ch_2%2)*2 + (ch_3%2)*4) * 4  # width bias
                            for h in range(7):
                                for w in range(7):
                                    h_min = max(h * 32 + h_bias, 0)
                                    h_max = min((h+1) * 32 + h_bias, 224)
                                    w_min = max(w * 32 + w_bias, 0)
                                    w_max = min((w+1) * 32 + w_bias, 224)
                                    # update reconstruction max and min by output
                                    recon_max[color, h_min:h_max, w_min:w_max] = torch.min(recon_max[color, h_min:h_max, w_min:w_max].data, \
                                                torch.ones_like(recon_max[color, h_min:h_max, w_min:w_max]) * out[ch, h, w])
                                    recon_min[color, h_min:h_max, w_min:w_max] = torch.max(recon_min[color, h_min:h_max, w_min:w_max].data, \
                                                torch.ones_like(recon_min[color, h_min:h_max, w_min:w_max]) * (1-out[ch+3, h, w]))
            # reconstruct by the average of max and min
            recon_in = (recon_max.data + recon_min.data) / 2
            # smooth filtering
            recon_in = self.smooth_filter(recon_in.reshape(1,3,224,224))[0]
            # clip the reconstruction
            recon_in = torch.clamp(recon_in, min=0, max=1)
            return recon_in


    def reconstruction(self, inputs, labels, class_names, print_class=False):
        # reconstruct images of selected classes with valid gradient
        with torch.no_grad():
            # performance measurement
            recon_dict = {}
            max_ssim, sum_ssim = torch.tensor(0, dtype=torch.float32, requires_grad=False), torch.tensor(0, dtype=torch.float32, requires_grad=False)
            min_mse, sum_mse = torch.tensor(1, dtype=torch.float32, requires_grad=False), torch.tensor(0, dtype=torch.float32, requires_grad=False)
            max_psnr, sum_psnr = torch.tensor(0, dtype=torch.float32, requires_grad=False), torch.tensor(0, dtype=torch.float32, requires_grad=False)
            # check valid classes via gradient
            dense_grads = []
            for para in self.dense1[0].parameters():
                dense_grads.append(para.grad)

            # reconstruct and evaluate
            c_list = [*set(labels.tolist())] if self.grad_noise else torch.where(dense_grads[1]<-1e-6)[0].tolist()
            for c in c_list:
                if c >= self.num_classes:
                    break  # If noise exists, stop reconstructing nodes out of the range of 'ones' dense parameters
                recon_in = self.reconstruct_single_class(c=c, dense_grads=dense_grads)
                recon_dict[c] = recon_in
                for i in (labels==c).nonzero():
                    if self.evaluate:
                        origs = torch.reshape(inputs[i[0],:,:,:], (1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))
                        preds = torch.reshape(recon_in, (1,recon_in.shape[0],recon_in.shape[1],recon_in.shape[2]))
                        ssim = self.ssim(preds, origs)
                        max_ssim = max(ssim, max_ssim)
                        sum_ssim += ssim
                        mse = self.mse(preds, origs)
                        min_mse = min(mse, min_mse)
                        sum_mse += mse
                        psnr = self.psnr(preds, origs)
                        max_psnr = max(psnr, max_psnr)
                        sum_psnr += psnr
                # print(recon_in)
                # print(ssim, mse, psnr)
                # plt.imshow(recon_in.permute(1,2,0))
                # plt.show()
                # plt.close()

            # plot all results
            # plot input
            grid_shape = int(torch.as_tensor(inputs.shape[0]).sqrt().ceil())
            s = 12 if inputs.shape[3] > 150 else 6
            if torch.as_tensor(inputs.shape[0]).sqrt().item().is_integer():
                fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            else:
                fig, axes = plt.subplots(1, int(torch.as_tensor(inputs.shape[0])), figsize=(s, 1))
            label_classes = []
            for i, (im, axis) in enumerate(zip(inputs, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if labels is not None:
                    label_classes.append(class_names[labels[i]])
                axis.axis("off")
            plt.savefig('images/input_sample_all.png')
            plt.close()
            # plot reconstruction
            if torch.as_tensor(inputs.shape[0]).sqrt().item().is_integer():
                fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            else:
                fig, axes = plt.subplots(1, int(torch.as_tensor(inputs.shape[0])), figsize=(s, 1))
            label_classes = []
            for i, axis in enumerate(axes.flatten()):
                # print('i', i, 'labels[i].item()', labels[i].item())
                # print('recon_dict[labels[i].item()] min, max', torch.min(recon_dict[labels[i].item()]), torch.max(recon_dict[labels[i].item()]))
                # print('recon_dict[labels[i].item()]', recon_dict[labels[i].item()])
                axis.imshow(recon_dict[labels[i].item()].permute(1, 2, 0).cpu())
                if labels is not None:
                    label_classes.append(class_names[labels[i]])
                axis.axis("off")
            plt.savefig('images/reconstruct_all.png')
            plt.close()
            if print_class:
                print(label_classes)
        if self.evaluate:
            print('max ssim: {:.4f}; avg ssim: {:.4f}; min mse: {:.4f}; avg mse: {:.4f}; max psnr: {:.4f}; avg psnr: {:.4f}'.format(max_ssim.item(),
                        (sum_ssim / inputs.shape[0]).item(), min_mse.item(), (sum_mse / inputs.shape[0]).item(), max_psnr.item(), (sum_psnr / inputs.shape[0]).item()))
            return max_ssim.item(), (sum_ssim / inputs.shape[0]).item(), max_psnr.item(), (sum_psnr / inputs.shape[0]).item()



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    # import warnings
    # warnings.filterwarnings("ignore", module="matplotlib\..*")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', DEVICE)

    custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    batch_size = 100
    dataset = datasets.CIFAR100(root='data', train=False, transform=custom_transform, download=True)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataset_len = len(dataset)

    net = VGG16().to(DEVICE)

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        print('inputs.shape:', inputs.shape, 'labels.shape:', labels.shape)
        # print('inputs.shape', inputs.shape)  # torch.Size([10, 3, 224, 224])
        print('sorted labels:', labels.sort()[0])
        # Get gradient
        net.process(inputs, labels, DEVICE)
        # reconstruct input
        net.reconstruction(inputs, labels, dataset.classes)
        break

