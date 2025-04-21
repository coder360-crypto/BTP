"""Utility functions for image processing in Medical VQA."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
import os

from src.config import BLUR_SIGMA, SAVE_VISUALIZATIONS, VISUALIZATION_DIR


def load_image(image_path: str) -> np.ndarray:
    """Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_roi_mask(
    image: np.ndarray, 
    roi_coords: List[Tuple[int, int, int, int]], 
    blur_sigma: int = BLUR_SIGMA
) -> np.ndarray:
    """Apply ROI mask to focus on specific regions and blur the rest.
    
    Args:
        image: Input image
        roi_coords: List of ROI coordinates as (x1, y1, x2, y2) tuples
        blur_sigma: Sigma for Gaussian blur
        
    Returns:
        Image with ROIs highlighted and rest blurred
    """
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for all ROIs
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    
    for x1, y1, x2, y2 in roi_coords:
        mask[y1:y2, x1:x2] = 255
    
    # Ensure blur_sigma is positive and odd
    if blur_sigma <= 0:
        blur_sigma = 21  # Default to 21 if invalid
    if blur_sigma % 2 == 0:
        blur_sigma += 1  # Make it odd
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(image, (blur_sigma, blur_sigma), 0)
    
    # Create the final image: ROI from original, rest from blurred
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    result = image * mask_3d + blurred * (1 - mask_3d)
    
    return result.astype(np.uint8)


def draw_roi_boxes(
    image: np.ndarray, 
    roi_coords: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None
) -> np.ndarray:
    """Draw bounding boxes around regions of interest.
    
    Args:
        image: Input image
        roi_coords: List of ROI coordinates as (x1, y1, x2, y2) tuples
        scores: Optional list of confidence scores for each ROI
        
    Returns:
        Image with ROI bounding boxes drawn
    """
    result = image.copy()
    
    for idx, (x1, y1, x2, y2) in enumerate(roi_coords):
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add score if provided
        if scores and idx < len(scores):
            score_text = f"{scores[idx]:.2f}"
            cv2.putText(
                result, score_text, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
    
    return result


def save_visualization(
    image: np.ndarray, 
    filename: str, 
    question: Optional[str] = None
) -> str:
    """Save visualization image with optional question overlay.
    
    Args:
        image: Image to save
        filename: Output filename
        question: Optional question to overlay on the image
        
    Returns:
        Path to the saved visualization
    """
    if not SAVE_VISUALIZATIONS:
        return ""
    
    # Create matplotlib figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Add question as title if provided
    if question:
        plt.title(question, wrap=True, fontsize=12)
    
    plt.axis('off')
    
    # Ensure filename has proper extension
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        filename += '.png'
    
    # Full path for saving
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path 