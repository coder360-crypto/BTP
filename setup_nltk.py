#!/usr/bin/env python
"""Setup script to download all required NLTK resources."""

import os
import nltk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_nltk_resources():
    """Download all required NLTK resources."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Set the NLTK data path to include our directory
    nltk.data.path.insert(0, data_dir)
    
    # List of resources to download
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    # Download each resource
    for resource in resources:
        logger.info(f"Downloading NLTK resource: {resource}")
        try:
            nltk.download(resource, download_dir=data_dir, quiet=False)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Error downloading {resource}: {e}")
    
    # Verify the resources were downloaded
    for resource in resources:
        try:
            nltk.data.find(f"{resource}")
            logger.info(f"Verified resource exists: {resource}")
        except LookupError:
            logger.error(f"Resource {resource} not found after download attempt")
    
    # Special check for punkt_tab which is part of punkt
    try:
        nltk.data.find("tokenizers/punkt_tab")
        logger.info("Verified punkt_tab exists")
    except LookupError:
        logger.warning("punkt_tab not found - punkt may not be properly installed")
        # Try one more time with explicit path
        try:
            punct_dir = os.path.join(data_dir, 'tokenizers', 'punkt')
            if os.path.exists(punct_dir):
                logger.info(f"Found punkt directory at: {punct_dir}")
                files = os.listdir(punct_dir)
                logger.info(f"Files in punkt directory: {files}")
        except Exception as e:
            logger.error(f"Error checking punkt directory: {e}")

if __name__ == "__main__":
    logger.info("Setting up NLTK resources...")
    setup_nltk_resources()
    logger.info("NLTK setup complete") 