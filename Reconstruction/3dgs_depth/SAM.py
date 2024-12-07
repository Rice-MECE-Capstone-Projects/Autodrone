import os

from tqdm import tqdm

import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import time
from path import Path

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(num_masks + 1, 1 , figsize=(8, 15))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")

def main():
    warnings.filterwarnings("ignore")

    folder_path = "Dataset/rgbd_dataset_freiburg1_desk2/images"
    masks_path = folder_path + "/masks"
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    image_dir = Path(folder_path)
    files = sum([image_dir.files('*.{}'.format(ext)) for ext in ['png', 'jpg', 'bmp', 'ppm', 'JPG']], [])
    files.sort()

    text_prompt = "computer" # specify your text prompt here
    try:
        model = LangSAM()
        for file in tqdm(files): # for each image in the folder path
            file_name = os.path.splitext(os.path.basename(file))[0]
            ext = os.path.splitext(os.path.basename(file))[1]
            image_pil = Image.open(file).convert("RGB")

            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

            if len(masks) == 0:
                print(f"No objects of the '{text_prompt}' prompt detected in the image.")
                # Todo: add empty mask
                image_size = (1, image_pil.size[1], image_pil.size[0])
                masks_np = np.full(image_size, False, dtype=bool)
            else:
                # Convert masks to numpy arrays
                masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

            # Save the masks
            # mask_path = masks_path + "/{}.png".format(file_name + ext) # for colmap
            mask_path = masks_path + "/{}_mask.png".format(file_name) # for 3dgs
            merged_mask = np.logical_or.reduce(masks_np)
            save_mask(merged_mask, mask_path)

            # Print the bounding boxes, phrases, and logits
            print_bounding_boxes(boxes)
            print_detected_phrases(phrases)
            print_logits(logits)

    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
