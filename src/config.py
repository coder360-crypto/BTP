"""Configuration module for Medical VQA system."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "gpt-4-vision-preview")
VLM_TEMP = float(os.environ.get("VLM_TEMP", "0.2"))

# Image processing
ROI_DETECTION_ENABLED = os.environ.get("ROI_DETECTION_ENABLED", "False").lower() == "false"
ROI_DETECTION_THRESHOLD = float(os.environ.get("ROI_DETECTION_THRESHOLD", "0.15"))
# Blur kernel size must be positive and odd for OpenCV's GaussianBlur
BLUR_SIGMA = int(os.environ.get("BLUR_SIGMA", "21"))
if BLUR_SIGMA % 2 == 0:
    BLUR_SIGMA += 1  # Make it odd if even

# Graph parameters
MIN_GROUNDING_SCORE = float(os.environ.get("MIN_GROUNDING_SCORE", "0.65"))
NODE_SIMILARITY_THRESHOLD = float(os.environ.get("NODE_SIMILARITY_THRESHOLD", "0.75"))

# Visualization options
SAVE_VISUALIZATIONS = os.environ.get("SAVE_VISUALIZATIONS", "True").lower() == "true"
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR", "./outputs/visualizations")

# Ensure visualization directory exists
if SAVE_VISUALIZATIONS and not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Prompt templates
CONCEPT_EXTRACTION_PROMPT = """
You are an emergency room physician analyzing an image related to an emergency room scenario.
Focus on what is clearly visible in the image and describe:

1. The overall scene setting (hospital room, ambulance, trauma bay, etc.)
2. People present in the scene (healthcare providers, patients, etc.)
3. Actions or procedures being performed, if any
4. Equipment or tools visible in the scene
5. Patient condition or clinical elements if relevant
6. Spatial relationships between elements in the scene

Output your observations in this structured JSON format:
{
  "scene_setting": [
    {"name": "setting_name", "attributes": {"location": "", "appearance": ""}}
  ],
  "personnel": [
    {"name": "role_title", "attributes": {"actions": "", "position": ""}}
  ],
  "procedures": [
    {"name": "procedure_name", "attributes": {"technique": "", "stage": ""}}
  ],
  "equipment": [
    {"name": "equipment_name", "attributes": {"usage": "", "position": ""}}
  ],
  "clinical_elements": [
    {"name": "element_name", "attributes": {"significance": "", "appearance": ""}}
  ],
  "relationships": [
    {"subject": "entity1", "relation": "relation_type", "object": "entity2"}
  ]
}

Only include information you can confidently identify in the image.
Do not assume medical conditions or diagnoses that aren't clearly evident.
If the image is not from an emergency room context, describe the general scene without forcing medical terminology.
"""

ANSWER_SYNTHESIS_PROMPT = """
Based only on the provided context from the image analysis, answer the following question about the emergency room or medical scene.
If the context does not contain the information needed to answer, clearly state what is visible in the image without speculating.
Do not include any information that is not supported by the provided context.

Question: {question}

Context:
{context}

Answer:
""" 