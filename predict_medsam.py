import torch
import numpy as np
import pandas as pd
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os
from tqdm import tqdm # A nice progress bar

# --- 1. SET YOUR PARAMETERS HERE ---

MODEL_CHECKPOINT = "work_dir/SAM/sam_vit_h_4b8939.pth" 
DATA_DIR = "original_images/train/" # <-- CHANGE THIS PATH
ANNOTATION_FILENAME = "_annotations.csv" # <-- CHANGE THIS if your filename is different
OUTPUT_DIR = "medsam_results_cleaned/"
DEVICE = "mps" 

# --- 2. SETUP ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading MedSAM model...")
sam = sam_model_registry["vit_h"](checkpoint=MODEL_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print("Model loaded.")

annotation_path = os.path.join(DATA_DIR, ANNOTATION_FILENAME)
print(f"Loading annotations from {annotation_path}...")
df = pd.read_csv(annotation_path)
image_groups = df.groupby('filename')
print(f"Found annotations for {len(image_groups)} unique images.")

# --- 3. LOOP, PREDICT, AND SAVE ---

for filename, group in tqdm(image_groups, desc="Processing Images"):
    image_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found, skipping: {filename}")
        continue
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    # --- MODIFIED PART: Loop through each box individually ---
    
    # Create a list to hold the masks for each box in this image
    all_masks_for_image = []
    
    # Get all bounding boxes for this image
    input_boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values
    
    # Loop through each bounding box one by one
    for box in input_boxes:
        # Run prediction for a SINGLE bounding box
        masks, scores, logits = predictor.predict(
            box=box[None, :], # Add a batch dimension for a single box
            multimask_output=False,
        )
        # Add the resulting mask to our list
        all_masks_for_image.append(masks[0])

    # --- END MODIFIED PART ---
    
    # Combine the multiple masks into a single final mask
    if all_masks_for_image:
        final_mask = np.any(np.array(all_masks_for_image), axis=0)
    else:
        # Create an empty mask if there were no annotations for this image
        final_mask = np.zeros((image.shape[0], image.shape[1]))

    # Post-process the mask to remove noise
    final_mask_img = final_mask.astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(final_mask_img, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Save the CLEANED final mask
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, cleaned_mask)

print(f"\nPrediction complete! All cleaned masks saved in: {OUTPUT_DIR}")




















import tensorflow.keras.backend as K
from tensorflow.keras import saving

@saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

@saving.register_keras_serializable()
def weighted_bce_dice_loss(y_true, y_pred):
    # We are giving BCE loss more weight to emphasize per-pixel correctness
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.8 * bce + 0.2 * dice