# 3D reconstruction
**The code and document is based on original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)**, and installation and usage are similar. Please see their repo for more details. Below is a simplified and modified version based on their original readme file, which should work for most cases. We thank 3DGS team for their excellent work!

We add two interesting features to the 3D gaussian splatting reconstruction. First we use depth ground truth for supervision because our auto-drone will be able to capture depth in the future. Second, we import semantic masks for object extracton.

## Requirements
- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- We implement the code in Ubuntu 22.04

## Easy Setup

Install based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_autodrone
```
Please refer to [COLMAP website](https://github.com/colmap/colmap) for COLMAP installation. We suggest install the GPU version. 

## Dataset preparing
### Mask Preparation
We use [language_SAM](https://github.com/luca-medeiros/lang-segment-anything) for masks generation, which allows semantic segmentation given test prompts. Please refer to their implementation for installation. It is suggested to create another environment for this task. 

Here we provide ```SAM.py``` for convenient mask generation and renaming. The generated masks will be available for 3dgs masking. To first use these masks for COLMAP initialization, please use ```maskColmap.py``` to convert color and rename files.
### COLMAP Initialization
To train 3DGS with object masking, you need to mask out unwanted key points during structure-from-motion process. You can use the provided ```convert_mask.py``` script to do that. The expected data structure is as follows:
```
<location>
|---input
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---masks_colmap
    |---<image_0.png.png>
    |---<image_1.png.png>
    |---...
```
Then run the script
```shell
python convert_mask.py -s <dataset_location>
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert_mask.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
</details>
<br>

If masks are not provided, all key points detected will be used for matching and initialization.

### Data structure ready for rasterization
Based on the original COLMAP loaders, we further load ground truth depth and masks for supervision and object extraction, which expects the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---masks
|   |---<image_0_mask.png>
|   |---<image_1_mask.png>
|   |---...
|---depth
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```
Masks and depth ground truth are not necessary to run the code. If not provided, the pipeline will not do masking during training or will not optimize with depth loss.

## Running

To train the 3DGS model:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interal
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.1``` by default.

</details>
<br>


## Evaluation (same as original 3DGS)
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

### Visualization
You can follow the original 3DGS [Interactive Viewr](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) to visualize the results or manually use the render function and save the render results such as:
```python
rendering = render(view, gaussians, pipeline, background, masks=view.mask)["render"]
torchvision.utils.save_image(rendering, "render.png")
```
you can visualize the depth with:
```python
rendering = render(view, gaussians, pipeline, background, masks=view.mask)["depth"]
rendering = (rendering - rendering.min()) / (rendering.max() - rendering.min())
torchvision.utils.save_image(rendering, "depth.png")
```