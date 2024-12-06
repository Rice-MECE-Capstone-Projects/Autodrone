import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import time

from depth_anything_v2.dpt import DepthAnythingV2
#from Depth_Estimator_594.depth_anything_v2.dpt import DepthAnythingV2

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def set_axes_equal(ax):
    """Set equal aspect ratio for 3D axes to make units appear equally scaled."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle, y_middle, z_middle = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range)
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([0, 255])  # Set depth value limits

def load_model(encoder_type='vits'):
    """Initialize and load DepthAnythingV2 model."""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**model_configs[encoder_type])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    #model.load_state_dict(torch.load(f'Depth_Estimator_594/depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    return model.to(DEVICE).eval()

def normalize_depth(depth):
    """Normalize depth values to 0-255 range for saving as an image."""
    Z_min, Z_max = depth.min(), depth.max()
    Z_normalized = (depth - Z_min) / (Z_max - Z_min) * 255
    return Z_normalized.astype(np.uint8)

def plot_results(raw_img_rgb, Z_normalized, X, Y, Z):
    """Plot original image, depth map, and 3D surface depth map."""
    fig = plt.figure(figsize=(18, 6))

    # Original Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(raw_img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Depth Map
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(Z_normalized, cmap='gray')
    ax2.set_title('Depth Map')
    ax2.axis('off')

    # 3D Surface Plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    stride = 8  # Adjust stride for performance
    X_down, Y_down, Z_down = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride]
    ax3.plot_surface(X_down, Y_down, Z_down, rstride=1, cstride=1, facecolors=raw_img_rgb[::stride, ::stride] / 255.0, edgecolor='none', alpha=0.8)

    ax3.set_title('3D Surface Depth Map')
    ax3.set_xlabel('Width (W)')
    ax3.set_ylabel('Height (H)')
    ax3.set_zlabel('Depth Value')
    set_axes_equal(ax3)
    plt.tight_layout()
    plt.show()

def main(args):
    full_start = time.time()
    # Load model
    model = load_model(args.encoder)

    # Load and prepare the image
    raw_img = cv2.imread(args.image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Image not found at path: {args.image_path}")
    raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    print(f"Original Image Shape: {raw_img.shape}")

    # Infer depth
    start_time = time.time()
    with torch.no_grad():
        depth = model.infer_image(raw_img)
    elapsed_time = time.time() - start_time
    print(f"Depth inference took {elapsed_time:.4f} seconds.")
    print(f"Depth Map Shape: {depth.shape}, Mean Depth: {depth.mean():.2f}")

    # Normalize depth map for saving
    Z_normalized = normalize_depth(depth)
    depth_image = Image.fromarray(Z_normalized)
    depth_image.save('depth_map.png')
    print("Depth map saved as 'depth_map.png'.")

    # Prepare data for 3D plotting
    H, W = depth.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    Z = 255 - Z_normalized  # Invert for 3D visualization

    #plot_results(raw_img_rgb, Z_normalized, X, Y, Z)
    full_end = time.time()

    full_elapsed = full_end - full_start
    print(f"Full time took {full_elapsed:.4f} seconds.")
    return depth.mean()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth map inference and visualization.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help="Model encoder type.")
    
    args = parser.parse_args()
    main(args)
 
    #args = ['--image_path', '../ryon.jpeg', '--encoder', 'vits']
    #main(args)
