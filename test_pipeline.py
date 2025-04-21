#!/usr/bin/env python
"""Test script for the Medical VQA pipeline."""

import os
import logging
import argparse
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from statistics import mean, median, mode
from collections import defaultdict

from src.core.vqa_pipeline import MedicalVQAPipeline
from src.utils.image_utils import load_image
from src.utils.nlp_utils import calculate_all_metrics

# Define constant model settings
GROQ_API_KEY = "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"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ROI_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    # To run this file with CSV batch processing and limited samples:
    #   python test_pipeline.py --csv ervqa_data/test_vqa_v4.csv --images-dir ervqa_data/test_v2_images/ --limit 5
    
    parser = argparse.ArgumentParser(description="Test Medical VQA pipeline")
    parser.add_argument("--output-dir", type=str, default="./test_outputs",
                        help="Directory to save test results")
    parser.add_argument("--mock", action="store_true", 
                        help="Run in mock mode without making actual API calls")
    parser.add_argument("--enable-grounding", action="store_true",
                        help="Enable grounding score calculation (default: off)")
    parser.add_argument("--enable-roi-detection", action="store_true",
                        help="Enable ROI detection (default: off, overrides config)")
    parser.add_argument("--disable-roi-detection", action="store_true",
                        help="Disable ROI detection (default: off, overrides config)")
    parser.add_argument("--csv", type=str, default="ervqa_data/test_vqa_v4.csv",
                        help="Path to CSV file with image paths, questions, and ground truth answers")
    parser.add_argument("--images-dir", type=str, default="ervqa_data/test_v2_images/",
                        help="Directory containing the test images")
    parser.add_argument("--limit", type=int, default=5,
                        help="Limit the number of samples to process from the CSV (default: 5)")
    
    return parser.parse_args()


def get_test_questions():
    """Get a list of test questions for the pipeline."""
    return [
        "What injuries are visible in this image?",
        "Is the patient's condition critical?",
        "What medical equipment is being used?",
        "Is proper medical procedure being followed?",
        "What anatomical structures are visible?"
    ]


def generate_metric_graphs(metrics_data, output_dir):
    """Generate graphs for the evaluation metrics.
    
    Args:
        metrics_data: Dictionary with metric names as keys and list of values as values
        output_dir: Directory to save the generated graphs
    """
    # Create graphs directory
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Define metrics and their labels
    metrics = ['bleu', 'rouge', 'sent', 'ent']
    if 'clip_c' in metrics_data and any(metrics_data['clip_c']):
        metrics.append('clip_c')
    
    metric_labels = {
        'bleu': 'BLEU-1 Score',
        'rouge': 'ROUGE-L Score',
        'sent': 'SentenceBERT Similarity',
        'ent': 'Entailment Score',
        'clip_c': 'CLIP-Score Confidence'
    }
    
    # Calculate statistical measures for each metric
    stats = {}
    for metric in metrics:
        values = [v for v in metrics_data[metric] if v is not None]
        if values:
            stats[metric] = {
                'mean': mean(values),
                'median': median(values),
                'min': min(values),
                'max': max(values),
                'std': np.std(values)
            }
    
    # 1. Comprehensive box plot for all metrics
    plt.figure(figsize=(12, 8))
    data_to_plot = []
    labels = []
    
    for metric in metrics:
        if metric in metrics_data and metrics_data[metric]:
            values = [v for v in metrics_data[metric] if v is not None]
            if values:
                data_to_plot.append(values)
                labels.append(metric_labels[metric])
    
    if data_to_plot:
        # Create box plot with enhanced styling
        box_plot = plt.boxplot(data_to_plot, patch_artist=True, 
                             labels=labels, showmeans=True, 
                             meanprops={"marker":"o", "markerfacecolor":"white", 
                                       "markeredgecolor":"black", "markersize":8})
        
        # Set colors for box plots
        colors = ['lightblue', 'lightgreen', 'salmon', 'lightyellow', 'lightcyan']
        for patch, color in zip(box_plot['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        # Add individual data points (jittered to avoid overlap)
        for i, d in enumerate(data_to_plot):
            # Add jitter to x position
            x = np.random.normal(i+1, 0.04, size=len(d))
            plt.scatter(x, d, alpha=0.4, s=15, c='darkblue', edgecolor='none')
        
        # Add grid, title, and labels with improved styling
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title("Comparison of Evaluation Metrics", fontsize=16, pad=20)
        plt.ylabel("Score", fontsize=14)
        plt.ylim(0, 1.05)  # Scores are between 0 and 1
        
        # Add mean and median labels to the box plot
        for i, metric in enumerate(metrics):
            if metric in stats:
                mean_val = stats[metric]['mean']
                median_val = stats[metric]['median']
                plt.annotate(f"Mean: {mean_val:.3f}", xy=(i+1, mean_val), 
                           xytext=(i+1.15, mean_val),
                           fontsize=8, va='center')
                plt.annotate(f"Median: {median_val:.3f}", xy=(i+1, median_val), 
                           xytext=(i+1.15, median_val-0.02),
                           fontsize=8, va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "metrics_boxplot.png"), dpi=300)
        plt.close()
    
    # 2. Individual distribution plots with KDE for each metric
    for metric in metrics:
        if metric not in metrics_data or not metrics_data[metric]:
            continue
            
        values = [v for v in metrics_data[metric] if v is not None]
        if not values or len(values) < 2:  # Need at least 2 points for KDE
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Create a more informative distribution plot
        # Histogram with KDE curve
        plt.hist(values, bins=min(10, len(values)), alpha=0.6, color='skyblue', 
                density=True, label='Distribution')
        
        # Try to add KDE curve if we have enough data points
        if len(values) > 3:
            try:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(values)
                x = np.linspace(min(values), max(values), 100)
                plt.plot(x, density(x), 'r-', lw=2, label='Density Estimation')
            except Exception as e:
                logger.warning(f"Could not generate KDE for {metric}: {e}")
        
        # Add mean and median lines
        plt.axvline(stats[metric]['mean'], color='red', linestyle='dashed', 
                   linewidth=1.5, label=f"Mean: {stats[metric]['mean']:.3f}")
        plt.axvline(stats[metric]['median'], color='green', linestyle='dashed', 
                   linewidth=1.5, label=f"Median: {stats[metric]['median']:.3f}")
        
        # Add data points as a rug plot at the bottom
        plt.plot(values, [0]*len(values), 'k|', ms=20, alpha=0.5)
        
        plt.title(f"Distribution of {metric_labels[metric]}", fontsize=14)
        plt.xlabel(f"{metric_labels[metric]}", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{metric}_distribution.png"), dpi=300)
        plt.close()
    
    # 3. Correlation matrix between metrics (if we have enough metrics)
    if len(metrics) > 1:
        # Prepare data for correlation analysis
        metric_values = {}
        for metric in metrics:
            if metric in metrics_data:
                values = [v for v in metrics_data[metric] if v is not None]
                if values:
                    metric_values[metric] = values
        
        # Check if we have enough common samples across metrics
        common_length = min(len(values) for values in metric_values.values()) if metric_values else 0
        
        if common_length > 1:  # Need at least 2 samples for correlation
            # Create correlation matrix
            corr_data = pd.DataFrame({metric: values[:common_length] 
                                    for metric, values in metric_values.items()})
            corr_matrix = corr_data.corr()
            
            plt.figure(figsize=(10, 8))
            # Create a heatmap for correlation
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='equal', 
                      vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                           ha="center", va="center", 
                           color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
            
            # Add labels and title
            plt.colorbar(label='Correlation Coefficient')
            plt.title("Correlation Between Metrics", fontsize=14)
            plt.xticks(range(len(corr_matrix.columns)), 
                      [metric_labels[m] for m in corr_matrix.columns], rotation=45)
            plt.yticks(range(len(corr_matrix.columns)), 
                      [metric_labels[m] for m in corr_matrix.columns])
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "metrics_correlation.png"), dpi=300)
            plt.close()
    
    # Save statistics as JSON
    with open(os.path.join(graphs_dir, "metrics_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Generated metric graphs in {graphs_dir}")


def run_mock_test(output_dir, enable_roi_detection=None):
    """Run tests with mock responses to verify pipeline structure."""
    logger.info("Running mock tests to validate pipeline structure")
    roi_status = "using config default" if enable_roi_detection is None else f"{'enabled' if enable_roi_detection else 'disabled'}"
    logger.info(f"ROI detection: {roi_status}")
    
    # Create test output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a default test image
    image_path = "ervqa_data/test_v2_images/headinjuriesinhospital3.jpeg"
    
    # Try to load the image to verify it exists
    try:
        image = load_image(image_path)
        logger.info(f"Successfully loaded test image: {image_path}")
    except Exception as e:
        logger.error(f"Failed to load test image: {e}")
        return
    
    # Mock results for pipeline components
    mock_roi_coords = [(50, 50, 150, 150)]
    
    mock_concepts = {
        "anatomical_structures": [
            {"name": "head", "attributes": {"location": "center", "appearance": "visible"}}
        ],
        "findings": [
            {"name": "laceration", "attributes": {"severity": "moderate", "appearance": "bleeding"}}
        ],
        "relationships": [
            {"subject": "laceration", "relation": "located_in", "object": "head"}
        ]
    }
    
    mock_grounded_concepts = mock_concepts.copy()
    for struct in mock_grounded_concepts["anatomical_structures"]:
        struct["grounding_score"] = 0.9
    for finding in mock_grounded_concepts["findings"]:
        finding["grounding_score"] = 0.8
    for rel in mock_grounded_concepts["relationships"]:
        rel["grounding_score"] = 0.7
    
    # Try to initialize the pipeline without making API calls
    # This will validate imports and class structure
    try:
        # Create dummy pipeline to test initialization with standard model parameters
        pipeline = MedicalVQAPipeline(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            roi_model=ROI_MODEL,
            enable_roi_detection=enable_roi_detection
        )
        logger.info("Successfully initialized pipeline")
        
        # Mock the pipeline components to avoid API calls
        pipeline.roi_detector.detect = lambda *args, **kwargs: mock_roi_coords
        pipeline.concept_extractor.extract_concepts = lambda *args, **kwargs: (mock_concepts, 0.9)
        pipeline.concept_extractor.check_grounding = lambda *args, **kwargs: mock_grounded_concepts
        
        # Test with one question
        test_questions = get_test_questions()
        test_question = test_questions[0]
        
        logger.info(f"Testing with question: {test_question}")
        
        # Create subdirectory for this test
        test_output_dir = os.path.join(output_dir, "mock_test")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Process the question
        result = pipeline.answer_question(image_path, test_question, test_output_dir)
        
        # Mock answer for testing
        mock_answer = "The image shows a patient with a laceration on the head with moderate bleeding."
        result['answer'] = mock_answer
        
        # Mock reference answer for testing
        mock_reference = "Patient has a head laceration with active bleeding at the site."
        
        # Add mock metrics
        mock_metrics = {
            'bleu': 0.75,
            'rouge': 0.62,
            'sent': 0.88,
            'ent': 0.72,
            'clip_c': 0.65
        }
        result['metrics'] = mock_metrics
        result['reference_answer'] = mock_reference
        
        # Save the result
        result_file = os.path.join(test_output_dir, "mock_test_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Mock test completed successfully. Results saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Error during pipeline initialization or test: {e}")


def run_csv_test(csv_path, images_dir, output_dir, enable_grounding=False, enable_roi_detection=False, limit=5):
    """Run tests using a CSV file with image paths, questions, and ground truth answers."""
    logger.info(f"Running tests on CSV dataset: {csv_path}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Limit: {limit} samples")
    logger.info(f"Grounding enabled: {enable_grounding}")
    roi_status = "using config default" if enable_roi_detection is None else f"{'enabled' if enable_roi_detection else 'disabled'}"
    logger.info(f"ROI detection: {roi_status}")
    
    # Create test output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # Apply limit to process only the specified number of samples
        df = df.head(limit)
        logger.info(f"Processing first {len(df)} samples")
        
        # Initialize the pipeline with standard model parameters
        pipeline = MedicalVQAPipeline(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            roi_model=ROI_MODEL,
            enable_grounding=enable_grounding,
            enable_roi_detection=enable_roi_detection
        )
        
        # Create results file for all metrics
        results_file = os.path.join(output_dir, "metrics_results.csv")
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "question", "reference_answer", "generated_answer", 
                "bleu", "rouge", "sent", "ent", "clip_c"
            ])
        
        # Collect metrics for later analysis
        all_metrics = defaultdict(list)
        processed_count = 0
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            image_filename = row['filename']
            question = row['question']
            reference_answer = row['answer']
            
            # Construct full image path
            image_path = os.path.join(images_dir, image_filename)
            
            # Verify image exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}, skipping")
                continue
            
            # Create subdirectory for this test
            sample_dir = os.path.join(output_dir, f"sample_{index}")
            os.makedirs(sample_dir, exist_ok=True)
            
            try:
                # Process the question
                logger.info(f"Processing {image_filename}: {question}")
                logger.info(f"Reference answer: {reference_answer}")
                result = pipeline.answer_question(image_path, question, sample_dir)
                
                # Get the generated answer
                generated_answer = result['answer']
                
                # Calculate all metrics
                metrics = calculate_all_metrics(reference_answer, generated_answer, image_path)
                
                # Add metrics to result
                result['metrics'] = metrics
                result['reference_answer'] = reference_answer
                
                # Collect metrics for later analysis
                for metric_name, metric_value in metrics.items():
                    all_metrics[metric_name].append(metric_value)
                
                # Save the complete result
                result_file = os.path.join(sample_dir, "result.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Append to metrics results file
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        image_filename, question, reference_answer, generated_answer,
                        metrics.get('bleu', 0), metrics.get('rouge', 0), 
                        metrics.get('sent', 0), metrics.get('ent', 0),
                        metrics.get('clip_c', 0)
                    ])
                
                # Update metrics
                processed_count += 1
                
                # Log metrics
                metrics_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"Metrics: {metrics_str}")
                
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
        
        # Calculate and save average statistics
        if processed_count > 0:
            # Generate metric graphs
            generate_metric_graphs(all_metrics, output_dir)
            
            # Calculate average metrics
            avg_metrics = {}
            for metric_name, values in all_metrics.items():
                filtered_values = [v for v in values if v is not None]
                if filtered_values:
                    avg_metrics[f"avg_{metric_name}"] = sum(filtered_values) / len(filtered_values)
                    avg_metrics[f"median_{metric_name}"] = sorted(filtered_values)[len(filtered_values) // 2]
                    avg_metrics[f"min_{metric_name}"] = min(filtered_values)
                    avg_metrics[f"max_{metric_name}"] = max(filtered_values)
            
            # Save summary
            summary_file = os.path.join(output_dir, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump({
                    "total_samples": len(df),
                    "processed_samples": processed_count,
                    "metrics": avg_metrics,
                    "enable_grounding": enable_grounding,
                    "roi_detection_enabled": enable_roi_detection,
                    "model": GROQ_MODEL,
                    "roi_model": ROI_MODEL
                }, f, indent=2)
            
            # Create a consolidated results file with all samples
            consolidated_results = []
            for index in range(len(df)):
                sample_dir = os.path.join(output_dir, f"sample_{index}")
                result_file = os.path.join(sample_dir, "result.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            sample_result = json.load(f)
                            consolidated_results.append(sample_result)
                    except Exception as e:
                        logger.error(f"Error loading result from {result_file}: {e}")
            
            # Save consolidated results
            consolidated_file = os.path.join(output_dir, "outputs.json")
            with open(consolidated_file, 'w') as f:
                json.dump(consolidated_results, f, indent=2)
        
        logger.info("CSV tests completed")
        
    except Exception as e:
        logger.error(f"Error during CSV testing: {e}")


def main():
    """Main function to run tests."""
    args = parse_args()
    
    logger.info("Starting Medical VQA pipeline tests")
    logger.info(f"Using models - Main: {GROQ_MODEL}, ROI: {ROI_MODEL}")
    
    # Determine ROI detection setting
    enable_roi_detection = False  # Default to using config value
    if args.enable_roi_detection and args.disable_roi_detection:
        logger.warning("Both --enable-roi-detection and --disable-roi-detection specified; using config default")
    elif args.enable_roi_detection:
        enable_roi_detection = True
        logger.info("ROI detection enabled via command line")
    elif args.disable_roi_detection:
        enable_roi_detection = False
        logger.info("ROI detection disabled via command line")
    
    if args.mock:
        # Run mock tests without API calls
        run_mock_test(args.output_dir, enable_roi_detection)
    else:
        # Run tests using CSV file with the first 5 samples by default
        run_csv_test(
            args.csv, 
            args.images_dir, 
            args.output_dir, 
            args.enable_grounding,
            enable_roi_detection,
            args.limit
        )


if __name__ == "__main__":
    main() 