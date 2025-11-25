import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image



def create_paired_downsample(img):
    channels = img.shape[1]
    filter_a = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter_a = filter_a.repeat(channels, 1, 1, 1)
    filter_b = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter_b = filter_b.repeat(channels, 1, 1, 1)
    result_a = torch.nn.functional.conv2d(img, filter_a, stride=2, groups=channels)
    result_b = torch.nn.functional.conv2d(img, filter_b, stride=2, groups=channels)
    return result_a, result_b

def gaussian_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))

def generate_gaussian_kernel(kernel_length=21, sigma=3, num_channels=1, device=None):
    step_size = (2 * sigma + 1.) / (kernel_length)
    x_coords = torch.linspace(-sigma - step_size / 2., sigma + step_size / 2., kernel_length + 1,)
    if device is not None:
        x_coords = x_coords.to(device)
    kernel_1d = torch.diff(gaussian_cdf(x_coords))
    kernel_2d = torch.sqrt(torch.outer(kernel_1d, kernel_1d))
    normalized_kernel = kernel_2d / torch.sum(kernel_2d)
    output_kernel = normalized_kernel.view(1, 1, kernel_length, kernel_length)
    output_kernel = output_kernel.repeat(num_channels, 1, 1, 1)
    return output_kernel

class LocalMean(torch.nn.Module):
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, img):
        img = torch.nn.functional.pad(img, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = img.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))

def apply_blur(x):
    device = x.device
    k_size = 21
    pad_size = k_size // 2
    gauss_kernel = generate_gaussian_kernel(k_size, 1, x.size(1), device=device)
    padded_x = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    return torch.nn.functional.conv2d(padded_x, gauss_kernel, padding=0, groups=x.size(1))

def add_padding(img):
    pad_val = 2
    pad_layer = torch.nn.ConstantPad2d(pad_val, 0)
    padded_img = pad_layer(img)
    return padded_img

def calculate_local_variance(noisy_tensor):
    batch, channels, width, height = noisy_tensor.shape
    avg_pooling = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    avg_noisy = avg_pooling(noisy_tensor)
    padded_avg = add_padding(avg_noisy)
    padded_noisy = add_padding(noisy_tensor)
    unfolded_avg = padded_avg.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy = padded_noisy.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_avg = unfolded_avg.reshape(unfolded_avg.shape[0], -1, 5, 5)
    unfolded_noisy = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    squared_diffs = (unfolded_noisy - unfolded_avg) ** 2
    variance = torch.mean(squared_diffs, dim=(2, 3))
    variance = variance.view(batch, channels, width, height)
    return variance

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6



def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def show_pic(pic, name, path):
    num_pics = len(pic)
    for i in range(num_pics):
        img = pic[i]
        img_array = img[0].cpu().float().numpy()
        if img_array.shape[0] == 3:
            img_array = (np.transpose(img_array, (1, 2, 0)))
            im = Image.fromarray(np.clip(img_array * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im)
        elif img_array.shape[0] == 1:
            im = Image.fromarray(np.clip(img_array[0] * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im, plt.cm.gray)
    plt.savefig(path)

pair_downsampler = create_paired_downsample
blur = apply_blur
