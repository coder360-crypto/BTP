"""Region of Interest (ROI) detection module based on perturbation analysis."""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import logging

from src.config import ROI_DETECTION_THRESHOLD
from src.utils.image_utils import load_image, draw_roi_boxes, save_visualization

logger = logging.getLogger(__name__)


class ROIDetector:
    """Detector for finding regions of interest in medical images using perturbation."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """Initialize the ROI detector.
        
        Args:
            model_name: Name of the vision-language model to use for perturbation analysis
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
               # _attn_implementation="flash_attention_2" if self.device.type == "cuda" else "eager"
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Grid parameters for perturbation
        self.grid_size = (4, 4)  # Default grid divisions (can be adjusted)
        self.threshold = ROI_DETECTION_THRESHOLD
        
        # Hyperparameter k for dynamic threshold calculation
        self.k = 4.0
    
    def detect(
        self, 
        image_path: str, 
        question: str, 
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """Detect regions of interest in the image relevant to the question.
        
        Args:
            image_path: Path to the medical image
            question: Medical question being asked
            grid_size: Optional custom grid size for perturbation analysis
            
        Returns:
            List of ROI coordinates as (x1, y1, x2, y2) tuples
        """
        # Update grid size if provided
        if grid_size is not None:
            self.grid_size = grid_size
        
        # Load image
        image = load_image(image_path)
        h, w = image.shape[:2]
        
        # Calculate baseline log probability
        baseline_prob = self._calculate_log_prob(image, question)
        
        # Calculate grid cell dimensions
        cell_h = h // self.grid_size[0]
        cell_w = w // self.grid_size[1]
        
        # Perturbation scores for each grid cell
        scores = np.zeros(self.grid_size)
        
        # Apply perturbation to each grid cell and measure effect
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Create a perturbed image with the current cell masked
                perturbed_image = image.copy()
                y_start, y_end = i * cell_h, min((i + 1) * cell_h, h)
                x_start, x_end = j * cell_w, min((j + 1) * cell_w, w)
                
                # Apply random noise perturbation to the cell
                mask = np.random.randint(0, 255, size=(y_end - y_start, x_end - x_start, 3), dtype=np.uint8)
                perturbed_image[y_start:y_end, x_start:x_end] = mask
                
                # Calculate log probability with perturbation
                perturbed_prob = self._calculate_log_prob(perturbed_image, question)
                
                # Calculate probability change (higher value means more important region)
                scores[i, j] = baseline_prob - perturbed_prob
        
        # Normalize scores
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        # Calculate dynamic threshold based on mean and standard deviation
        scores_flat = scores.flatten()
        mean_score = np.mean(scores_flat)
        std_score = np.std(scores_flat)
        
        # Dynamic threshold: mean - k * std (but ensure it's not negative)
        dynamic_threshold = max(0.0, mean_score - self.k * std_score)
        
        logger.info(f"Grounding scores - Mean: {mean_score:.4f}, Std: {std_score:.4f}")
        logger.info(f"Dynamic threshold (mean - {self.k} * std): {dynamic_threshold:.4f}")
        
        print(f"\n=== ROI GROUNDING SCORES ANALYSIS ===")
        print(f"Mean grounding score: {mean_score:.4f}")
        print(f"Standard deviation: {std_score:.4f}")
        print(f"Dynamic threshold (mean - {self.k} * std): {dynamic_threshold:.4f}")
        print(f"Individual cell scores: {scores_flat}")
        print("====================================\n")
        
        # Find grid cells that exceed the dynamic threshold
        roi_cells = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if scores[i, j] > dynamic_threshold:
                    y_start, y_end = i * cell_h, min((i + 1) * cell_h, h)
                    x_start, x_end = j * cell_w, min((j + 1) * cell_w, w)
                    roi_cells.append((x_start, y_start, x_end, y_end))
                    print(f"Accepted region ({i},{j}) with score {scores[i, j]:.4f} > {dynamic_threshold:.4f}")
                else:
                    print(f"Rejected region ({i},{j}) with score {scores[i, j]:.4f} <= {dynamic_threshold:.4f}")
        
        # Merge adjacent cells
        roi_coords = self._merge_adjacent_cells(roi_cells, scores.flatten())
        
        # Generate visualization if needed
        if roi_coords:
            vis_image = draw_roi_boxes(image, roi_coords)
            save_visualization(vis_image, f"roi_{image_path.split('/')[-1]}", question)
        
        return roi_coords
    
    def _calculate_log_prob(self, image: np.ndarray, question: str) -> float:
        """Calculate log probability of answer given image and question.
        
        Args:
            image: Image as numpy array
            question: Question text
            
        Returns:
            Log probability score
        """
        try:
            # Convert image to PIL
            image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create input messages in the chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
            ]
            
            # Prepare inputs
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image_pil], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extract log probabilities
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            
            # Return mean log probability
            return float(torch.mean(transition_scores).cpu().numpy())
        
        except Exception as e:
            logger.error(f"Error calculating log probability: {e}")
            return 0.0
    
    def _merge_adjacent_cells(
        self, 
        cells: List[Tuple[int, int, int, int]], 
        scores: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Merge adjacent cells with high scores into larger ROIs.
        
        Args:
            cells: List of cell coordinates (x1, y1, x2, y2)
            scores: Perturbation scores for cells
            
        Returns:
            List of merged ROI coordinates
        """
        if not cells:
            return []
        
        # Sort cells by score (highest first)
        sorted_indices = np.argsort(-scores[:len(cells)])
        sorted_cells = [cells[i] for i in sorted_indices]
        
        # Start with highest scoring cell
        merged_cells = [sorted_cells[0]]
        
        for cell in sorted_cells[1:]:
            merged = False
            
            # Try to merge with existing merged cells
            for i, merged_cell in enumerate(merged_cells):
                # Check if cells are adjacent or overlapping
                if (self._is_adjacent(cell, merged_cell) or self._is_overlapping(cell, merged_cell)):
                    # Merge cells
                    merged_cells[i] = (
                        min(cell[0], merged_cell[0]),
                        min(cell[1], merged_cell[1]),
                        max(cell[2], merged_cell[2]),
                        max(cell[3], merged_cell[3])
                    )
                    merged = True
                    break
            
            # If not merged, add as a new ROI
            if not merged:
                merged_cells.append(cell)
        
        return merged_cells
    
    def _is_adjacent(
        self, 
        cell1: Tuple[int, int, int, int], 
        cell2: Tuple[int, int, int, int], 
        threshold: int = 10
    ) -> bool:
        """Check if two cells are adjacent within a threshold.
        
        Args:
            cell1: First cell coordinates (x1, y1, x2, y2)
            cell2: Second cell coordinates (x1, y1, x2, y2)
            threshold: Maximum distance to be considered adjacent
            
        Returns:
            True if cells are adjacent, False otherwise
        """
        # Unpack coordinates
        x1_1, y1_1, x2_1, y2_1 = cell1
        x1_2, y1_2, x2_2, y2_2 = cell2
        
        # Check horizontal adjacency
        h_adjacent = (
            (abs(x1_1 - x2_2) <= threshold) or 
            (abs(x2_1 - x1_2) <= threshold)
        )
        
        # Check vertical adjacency
        v_adjacent = (
            (abs(y1_1 - y2_2) <= threshold) or 
            (abs(y2_1 - y1_2) <= threshold)
        )
        
        # Cells are adjacent if they are adjacent in one direction
        # and have overlap in the other direction
        h_overlap = (x1_1 <= x2_2 and x2_1 >= x1_2)
        v_overlap = (y1_1 <= y2_2 and y2_1 >= y1_2)
        
        return (h_adjacent and v_overlap) or (v_adjacent and h_overlap)
    
    def _is_overlapping(
        self, 
        cell1: Tuple[int, int, int, int], 
        cell2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two cells overlap.
        
        Args:
            cell1: First cell coordinates (x1, y1, x2, y2)
            cell2: Second cell coordinates (x1, y1, x2, y2)
            
        Returns:
            True if cells overlap, False otherwise
        """
        # Unpack coordinates
        x1_1, y1_1, x2_1, y2_1 = cell1
        x1_2, y1_2, x2_2, y2_2 = cell2
        
        # Check for overlap
        return (
            x1_1 < x2_2 and
            x2_1 > x1_2 and
            y1_1 < y2_2 and
            y2_1 > y1_2
        ) 