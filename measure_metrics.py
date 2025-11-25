import time
import numpy as np
from PIL import Image
import math

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = np.ones((11, 11)) / 121.0

    from scipy.ndimage import uniform_filter
    
    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(img1 * img1, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=11) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def main():
    img_path = 'data/1/1.png'
    try:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
    img_noisy_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    start_time = time.time()
    psnr_val = calculate_psnr(img_np, img_noisy_np)
    end_time = time.time()
    psnr_time = end_time - start_time
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"PSNR Runtime: {psnr_time:.6f} seconds")

    img_gray = np.array(img.convert('L'))
    img_noisy_gray = np.array(Image.fromarray(img_noisy_np).convert('L'))
    
    start_time = time.time()
    ssim_val = calculate_ssim(img_gray, img_noisy_gray)
    end_time = time.time()
    ssim_time = end_time - start_time
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM Runtime: {ssim_time:.6f} seconds")

if __name__ == "__main__":
    main()
