import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from utils import  pair_downsampler,calculate_local_variance,LocalMean

EPSILON = 1e-9
PI_CONST = 22.0 / 7.0


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smoothness_loss = SmoothLoss()
        self.texture_comparator = TextureDifference()
        self.local_averager = LocalMean(patch_size=5)
        self.tv_loss = L_TV()


    def forward(self, inp, pred1, pred2, denoised, illum, illum_d1, illum_d2, enhanced, enh_d1, enh_d2, cont3, ill3, cont4, ill4, final_cont, final_ill, pred3, pred4, tex_diff1, tex_diff2, blur_enh, blur_final):
        eps = 1e-9
        inp = inp + eps

        luminance = denoised.detach()[:, 2, :, :] * 0.299 + denoised.detach()[:, 1, :, :] * 0.587 + denoised.detach()[:, 0, :, :] * 0.144
        lum_mean = torch.mean(luminance, dim=(1, 2))
        enhance_factor = 0.5 / (lum_mean + eps)
        enhance_factor = enhance_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhance_factor = torch.clamp(enhance_factor, 1, 25)
        adjust_ratio = torch.pow(0.7, -enhance_factor) / enhance_factor
        adjust_ratio = adjust_ratio.repeat(1, 3, 1, 1)
        normalized = denoised.detach() / illum
        normalized = torch.clamp(normalized, eps, 0.8)
        enhanced_bright = torch.pow(denoised.detach() * enhance_factor, enhance_factor)
        clamped_bright = torch.clamp(enhanced_bright * adjust_ratio, eps, 1)
        clamped_adjusted = torch.clamp(denoised.detach() * enhance_factor, eps, 1)
        total_loss = 0
        total_loss += self.mse_loss(illum, clamped_bright) * 700
        total_loss += self.mse_loss(normalized, clamped_adjusted) * 1000
        total_loss += self.smoothness_loss(denoised.detach(), illum) * 5
        total_loss += self.tv_loss(illum) * 1600
        down1, down2 = pair_downsampler(inp)
        total_loss += self.mse_loss(down1, pred2) * 1000
        total_loss += self.mse_loss(down2, pred1) * 1000
        den1, den2 = pair_downsampler(denoised)
        total_loss += self.mse_loss(pred1, den1) * 1000
        total_loss += self.mse_loss(pred2, den2) * 1000
        total_loss += self.mse_loss(pred3, torch.cat([enh_d2.detach(), illum_d2.detach()], 1)) * 1000
        total_loss += self.mse_loss(pred4, torch.cat([enh_d1.detach(), illum_d1.detach()], 1)) * 1000
        final_d1, final_d2 = pair_downsampler(final_cont)
        total_loss += self.mse_loss(pred3[:, 0:3, :, :], final_d1) * 1000
        total_loss += self.mse_loss(pred4[:, 0:3, :, :], final_d2) * 1000
        total_loss += self.mse_loss(blur_enh.detach(), blur_final) * 10000
        total_loss += self.mse_loss(illum.detach(), final_ill) * 1000
        local_avg1 = self.local_averager(final_d1)
        local_avg2 = self.local_averager(final_d2)
        weighted1 = (1 - tex_diff2) * local_avg1 + final_d1 * tex_diff2
        weighted2 = (1 - tex_diff2) * local_avg2 + final_d1 * tex_diff2
        total_loss += self.mse_loss(final_d1, weighted1) * 10000
        total_loss += self.mse_loss(final_d2, weighted2) * 10000
        noise_var = calculate_local_variance(final_cont - enhanced)
        enhanced_var = calculate_local_variance(enhanced)
        total_loss += self.mse_loss(enhanced_var, noise_var) * 1000
        return total_loss

def create_gaussian_kernel(kernel_len=21, sigma=3, num_channels=1):
    step = (2 * sigma + 1.) / (kernel_len)
    x_vals = np.linspace(-sigma - step / 2., sigma + step / 2., kernel_len + 1)
    kernel_1d = np.diff(st.norm.cdf(x_vals))
    kernel_2d = np.sqrt(np.outer(kernel_1d, kernel_1d))
    normalized_kernel = kernel_2d / kernel_2d.sum()
    kernel_array = np.array(normalized_kernel, dtype=np.float32)
    kernel_array = kernel_array.reshape((kernel_len, kernel_len, 1, 1))
    kernel_array = np.repeat(kernel_array, num_channels, axis=2)

    return kernel_array


class TextureDifference(nn.Module):
    def __init__(self, patch_size=5, constant=1e-5, threshold=0.975):
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size
        self.constant = constant
        self.threshold = threshold

    def forward(self, img1, img2):
        img1 = self.convert_to_grayscale(img1)
        img2 = self.convert_to_grayscale(img2)

        std1 = self.compute_local_std(img1)
        std2 = self.compute_local_std(img2)
        num = 2 * std1 * std2
        den = std1 ** 2 + std2 ** 2 + self.constant
        similarity = num / den

        binary_mask = torch.where(similarity > self.threshold, torch.tensor(1.0, device=similarity.device),
                                  torch.tensor(0.0, device=similarity.device))

        return binary_mask

    def compute_local_std(self, img):
        pad_size = self.patch_size // 2
        img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        patches = img.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        patch_mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diffs = (patches - patch_mean) ** 2
        local_var = squared_diffs.mean(dim=(4, 5))
        local_std = torch.sqrt(local_var + 1e-9)
        return local_std

    def convert_to_grayscale(self, img):
        gray = 0.144 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.299 * img[:, 2, :, :]
        return gray.unsqueeze(1)


class L_TV(nn.Module):
    def __init__(self, weight=1):
        super(L_TV, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch = x.size()[0]
        height = x.size()[2]
        width = x.size()[3]
        h_count = (x.size()[2] - 1) * x.size()[3]
        w_count = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :height - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :width - 1]), 2).sum()
        return self.weight * 2 * (h_tv / h_count + w_tv / w_count) / batch

class Blur(nn.Module):
    def __init__(self, num_channels):
        super(Blur, self).__init__()
        self.num_channels = num_channels
        kernel = create_gaussian_kernel(kernlen=21, nsig=3, channels=self.num_channels)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.num_channels:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.num_channels))
        weight = self.weight.to(x.device)
        x = F.conv2d(x, weight, stride=1, padding=10, groups=self.num_channels)
        return x




class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def convert_rgb_to_ycbcr(self, img):

        flat_img = img.contiguous().view(-1, 3).float()
        device = img.device
        transform_mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).to(device)
        offset = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).to(device)
        transformed = flat_img.mm(transform_mat) + offset
        output = transformed.view(img.shape[0], 3, img.shape[2], img.shape[3])
        return output

    def forward(self, inp, out):


        self.output = out
        self.input = self.convert_rgb_to_ycbcr(inp)
        sigma_coeff = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_coeff)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                 keepdim=True) * sigma_coeff)
        p_norm = 1.0

        grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p_norm, dim=1, keepdim=True)
        grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p_norm, dim=1, keepdim=True)
        grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p_norm, dim=1, keepdim=True)
        grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p_norm, dim=1, keepdim=True)
        grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p_norm, dim=1, keepdim=True)
        grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p_norm, dim=1, keepdim=True)
        grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p_norm, dim=1, keepdim=True)
        grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p_norm, dim=1, keepdim=True)
        grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p_norm, dim=1, keepdim=True)
        grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p_norm, dim=1, keepdim=True)
        grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p_norm, dim=1, keepdim=True)
        grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p_norm, dim=1, keepdim=True)
        grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p_norm, dim=1,
                                        keepdim=True)
        grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p_norm, dim=1,
                                        keepdim=True)
        grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p_norm, dim=1,
                                        keepdim=True)
        grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p_norm, dim=1,
                                        keepdim=True)
        grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p_norm, dim=1,
                                        keepdim=True)
        grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p_norm, dim=1,
                                        keepdim=True)
        grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p_norm, dim=1,
                                        keepdim=True)
        grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p_norm, dim=1,
                                        keepdim=True)
        grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p_norm, dim=1,
                                        keepdim=True)
        grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p_norm, dim=1,
                                        keepdim=True)
        grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p_norm, dim=1,
                                        keepdim=True)
        grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p_norm, dim=1,
                                        keepdim=True)

        regularization_term = torch.mean(grad1) \
                    + torch.mean(grad2) \
                    + torch.mean(grad3) \
                    + torch.mean(grad4) \
                    + torch.mean(grad5) \
                    + torch.mean(grad6) \
                    + torch.mean(grad7) \
                    + torch.mean(grad8) \
                    + torch.mean(grad9) \
                    + torch.mean(grad10) \
                    + torch.mean(grad11) \
                    + torch.mean(grad12) \
                    + torch.mean(grad13) \
                    + torch.mean(grad14) \
                    + torch.mean(grad15) \
                    + torch.mean(grad16) \
                    + torch.mean(grad17) \
                    + torch.mean(grad18) \
                    + torch.mean(grad19) \
                    + torch.mean(grad20) \
                    + torch.mean(grad21) \
                    + torch.mean(grad22) \
                    + torch.mean(grad23) \
                    + torch.mean(grad24)

        return regularization_term
