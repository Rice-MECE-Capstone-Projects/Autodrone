from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImagePair(render_path, gt_path):
    render = Image.open(render_path)
    gt = Image.open(gt_path)
    render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
    return render_tensor, gt_tensor

def evaluate_single_image_pair(render_path, gt_path):
    render, gt = readImagePair(render_path, gt_path)

    ssim_val = ssim(render, gt).item()
    psnr_val = psnr(render, gt).item()
    lpips_val = lpips(render, gt, net_type='vgg').item()

    print("SSIM : {:>12.7f}".format(ssim_val))
    print("PSNR : {:>12.7f}".format(psnr_val))
    print("LPIPS: {:>12.7f}".format(lpips_val))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--render_path', '-r', required=True, type=str)
    parser.add_argument('--gt_path', '-g', required=True, type=str)
    args = parser.parse_args()
    evaluate_single_image_pair(args.render_path, args.gt_path)
