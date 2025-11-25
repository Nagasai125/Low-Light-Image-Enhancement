import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils import blur, pair_downsampler


class FirstStageDenoiser(nn.Module):
    def __init__(self, num_channels=48):
        super(FirstStageDenoiser, self).__init__()

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = nn.Conv2d(3, num_channels, 3, padding=1)
        self.layer2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.layer3 = nn.Conv2d(num_channels, 3, 1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


class SecondStageDenoiser(nn.Module):
    def __init__(self, num_channels=96):
        super(SecondStageDenoiser, self).__init__()

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = nn.Conv2d(6, num_channels, 3, padding=1)
        self.layer2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.layer3 = nn.Conv2d(num_channels, 6, 1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


class IlluminationEstimator(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(IlluminationEstimator, self).__init__()

        k_size = 3
        dil = 1
        pad = int((k_size - 1) / 2) * dil

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=k_size, stride=1, padding=pad),
            nn.ReLU()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=k_size, stride=1, padding=pad),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.residual_blocks.append(self.conv_block)

        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.input_conv(x)
        for block in self.residual_blocks:
            features = features + block(features)
        features = self.output_conv(features)
        features = torch.clamp(features, 0.0001, 1)

        return features


class MainNetwork(nn.Module):

    def __init__(self):
        super(MainNetwork, self).__init__()

        self.enhance = IlluminationEstimator(num_layers=3, num_channels=64)
        self.denoise_1 = FirstStageDenoiser(num_channels=48)
        self.denoise_2 = SecondStageDenoiser(num_channels=48)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.loss_calculator = LossFunction()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.texture_comparator = TextureDifference()


    def init_enhancer_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def init_denoiser_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, x):
        epsilon = 1e-4
        x = x + epsilon

        down1, down2 = pair_downsampler(x)
        pred1 = down1 - self.denoise_1(down1)
        pred2 = down2 - self.denoise_1(down2)
        denoised = x - self.denoise_1(x)
        denoised = torch.clamp(denoised, epsilon, 1)

        illum_map = self.enhance(denoised.detach())
        illum_down1, illum_down2 = pair_downsampler(illum_map)
        enhanced = x / illum_map
        enhanced = torch.clamp(enhanced, epsilon, 1)

        enhanced_down1 = down1 / illum_down1
        enhanced_down1 = torch.clamp(enhanced_down1, epsilon, 1)

        enhanced_down2 = down2 / illum_down2
        enhanced_down2 = torch.clamp(enhanced_down2, epsilon, 1)

        combined1 = torch.cat([enhanced_down1, illum_down1], 1).detach()
        pred3 = combined1 - self.denoise_2(combined1)
        pred3 = torch.clamp(pred3, epsilon, 1)
        content3 = pred3[:, :3, :, :]
        illum3 = pred3[:, 3:, :, :]

        combined2 = torch.cat([enhanced_down2, illum_down2], 1).detach()
        pred4 = combined2 - self.denoise_2(combined2)
        pred4 = torch.clamp(pred4, epsilon, 1)
        content4 = pred4[:, :3, :, :]
        illum4 = pred4[:, 3:, :, :]

        combined3 = torch.cat([enhanced, illum_map], 1).detach()
        pred5 = combined3 - self.denoise_2(combined3)
        pred5 = torch.clamp(pred5, epsilon, 1)
        final_content = pred5[:, :3, :, :]
        final_illum = pred5[:, 3:, :, :]

        texture_diff1 = self.texture_comparator(pred1, pred2)
        final_down1, final_down2 = pair_downsampler(final_content)
        texture_diff2 = self.texture_comparator(final_down1, final_down2)

        intermediate = denoised / illum_map
        intermediate = torch.clamp(intermediate, 0, 1)
        blurred_enhanced = blur(intermediate)
        blurred_final = blur(final_content)

        return pred1, pred2, denoised, illum_map, illum_down1, illum_down2, enhanced, enhanced_down1, enhanced_down2, content3, illum3, content4, illum4, final_content, final_illum, pred3, pred4, texture_diff1, texture_diff2, blurred_enhanced, blurred_final

    def compute_loss(self, x):
        pred1, pred2, denoised, illum_map, illum_down1, illum_down2, enhanced, enhanced_down1, enhanced_down2, content3, illum3, content4, illum4, final_content, final_illum, pred3, pred4, texture_diff1, texture_diff2, blurred_enhanced, blurred_final = self(x)
        total_loss = 0

        total_loss += self.loss_calculator(x, pred1, pred2, denoised, illum_map, illum_down1, illum_down2, enhanced, enhanced_down1, enhanced_down2, content3, illum3, content4, illum4, final_content, final_illum, pred3, pred4, texture_diff1, texture_diff2, blurred_enhanced, blurred_final)
        return total_loss


class InferenceModel(nn.Module):

    def __init__(self, weight_path, device=None):
        super(InferenceModel, self).__init__()

        self.enhance = IlluminationEstimator(num_layers=3, num_channels=64)
        self.denoise_1 = FirstStageDenoiser(num_channels=48)
        self.denoise_2 = SecondStageDenoiser(num_channels=48)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_weights = torch.load(weight_path, map_location=device)
        weight_dict = loaded_weights
        current_dict = self.state_dict()
        weight_dict = {k: v for k, v in weight_dict.items() if k in current_dict}
        current_dict.update(weight_dict)
        self.load_state_dict(current_dict)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, x):
        epsilon = 1e-4
        x = x + epsilon
        denoised = x - self.denoise_1(x)
        denoised = torch.clamp(denoised, epsilon, 1)
        illum_map = self.enhance(denoised)
        enhanced = x / illum_map
        enhanced = torch.clamp(enhanced, epsilon, 1)
        combined = torch.cat([enhanced, illum_map], 1).detach()
        pred = combined - self.denoise_2(combined)
        pred = torch.clamp(pred, epsilon, 1)
        final_output = pred[:, :3, :, :]
        return enhanced, final_output
