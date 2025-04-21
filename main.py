#!/usr/bin/env python
"""
Medical VQA with Focused Concept Graph and MI-Based Hallucination Control.

This script processes medical images and questions using a knowledge graph-based approach
with mechanistic interpretability checks to reduce hallucinations.
"""

import argparse
import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv

from src.core.vqa_pipeline import MedicalVQAPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medical_vqa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical VQA with hallucination mitigation")
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument("--image", type=str, help="Path to a single medical image")
    input_group.add_argument("--question", type=str, help="Question about the medical image")
    input_group.add_argument("--csv", type=str, help="Path to CSV file with image paths and questions")
    input_group.add_argument("--images-dir", type=str, default="./ervqa_data/test_v2_images/",
                            help="Directory containing images (used with --csv)")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--api-key", type=str, help="API key for VLM service")
    model_group.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                            help="Model identifier (default: google/gemma-3-27b-it)")
    model_group.add_argument("--roi-model", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct",
                           help="Model for ROI detection (default: HuggingFaceTB/SmolVLM-256M-Instruct)")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="./outputs",
                             help="Directory to save results (default: ./outputs)")
    output_group.add_argument("--save-interim", action="store_true",
                             help="Save interim results and visualizations")
    
    return parser.parse_args()


def process_single_image(pipeline, image_path, question, output_dir):
    """Process a single image and question."""
    logger.info(f"Processing image: {image_path}")
    logger.info(f"Question: {question}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Answer the question
    result = pipeline.answer_question(image_path, question, output_dir if args.save_interim else None)
    
    # Log the answer
    logger.info(f"Answer: {result['answer']}")
    
    # Save the result
    result_file = os.path.join(output_dir, f"result_{os.path.basename(image_path)}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def process_csv_file(pipeline, csv_path, images_dir, output_dir):
    """Process multiple image-question pairs from a CSV file."""
    logger.info(f"Processing questions from CSV: {csv_path}")
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} questions from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return []
    
    # Check required columns
    if "filename" not in df.columns or "question" not in df.columns:
        logger.error("CSV must contain 'filename' and 'question' columns")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each row
    results = []
    for i, row in df.iterrows():
        image_file = row["filename"]
        question = row["question"]
        
        # Construct image path
        image_path = os.path.join(images_dir, image_file)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Create output subdirectory for this image
        img_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
        
        try:
            # Process the image-question pair
            result = process_single_image(pipeline, image_path, question, img_output_dir)
            results.append(result)
            
            # Save the answer to the dataframe
            df.at[i, "generated_answer"] = result["answer"]
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    # Save the updated CSV with answers
    output_csv = os.path.join(output_dir, "results.csv")
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved results to {output_csv}")
    
    return results


def main(args):
    """Main function to run the Medical VQA pipeline."""
    # Initialize the pipeline
    pipeline = MedicalVQAPipeline(
        api_key=args.api_key,
        model=args.model,
        roi_model=args.roi_model
    )
    
    # Process based on input type
    if args.image and args.question:
        # Single image mode
        process_single_image(pipeline, args.image, args.question, args.output_dir)
    elif args.csv:
        # CSV batch mode
        process_csv_file(pipeline, args.csv, args.images_dir, args.output_dir)
    else:
        logger.error("Please provide either --image and --question, or --csv")


if __name__ == "__main__":
    args = parse_args()
    main(args) 