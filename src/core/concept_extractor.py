"""Concept extractor module for medical VQA."""

import json
import requests
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import time
import numpy as np
from PIL import Image
import io
import re  # Add missing import for regular expressions
from groq import Groq

from src.config import VLM_API_KEY, VLM_MODEL, VLM_TEMP, CONCEPT_EXTRACTION_PROMPT
from src.utils.image_utils import load_image

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extracts structured medical concepts from focused image regions using VLMs."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, enable_grounding: bool = False):
        """Initialize the concept extractor.
        
        Args:
            api_key: API key for the VLM service
            model: Model identifier for the VLM
            enable_grounding: Whether to perform grounding score calculation (default: False)
        """
        self.api_key = "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"
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.enable_grounding = enable_grounding
        
        # Check if required credentials are available
        if not self.api_key:
            logger.warning("API key not provided. Set VLM_API_KEY in .env or provide in constructor.")
        
        # Configure for provider
        self.provider = "groq"
        
    def _determine_provider(self) -> str:
        """Determine which VLM provider to use based on model name."""
        # Always use Groq API
        return "groq"
    
    def extract_concepts(
        self, 
        image: np.ndarray, 
        custom_prompt: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Extract structured medical concepts from the focused image.
        
        Args:
            image: The focused image as a numpy array
            custom_prompt: Optional custom prompt to override the default
            
        Returns:
            Tuple containing extracted concept dict and confidence score
        """
        # Convert numpy array to image bytes
        image_bytes = self._convert_image_to_bytes(image)
        
        # Get the appropriate prompt
        prompt = custom_prompt or CONCEPT_EXTRACTION_PROMPT
        
        # Call the Groq API
        concepts, confidence = self._call_groq_api(image_bytes, prompt)
        
        # Print the extracted concepts
        print("\n=== EXTRACTED CONCEPTS ===")
        print(json.dumps(concepts, indent=2))
        print(f"Confidence: {confidence}")
        print("=========================\n")
        
        return concepts, confidence
    
    def _convert_image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy image array to bytes for API transmission.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Image bytes
        """
        img_pil = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
    def _call_groq_api(
        self, 
        image_bytes: bytes, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Call Groq API with Meta's Llama model to extract concepts.
        
        Args:
            image_bytes: Image bytes
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        try:
            # Setup Groq client
            client = Groq(api_key=self.api_key)
            
            # Make API call
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=VLM_TEMP,
                max_completion_tokens=1000,
                top_p=0.7,
                stream=False
            )
            
            # Extract content and confidence
            content = response.choices[0].message.content
            confidence = 1.0 - VLM_TEMP  # Simple approximation
            
            # Parse JSON structure from response
            try:
                # Find JSON structure in the content (handles cases where model wraps JSON in markdown)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    concepts = json.loads(json_str)
                else:
                    logger.warning("No JSON structure found in VLM response")
                    concepts = {"error": "No structured data found in response", "raw_response": content}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                concepts = {"error": "Failed to parse structured data", "raw_response": content}
            
            return concepts, confidence
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return {"error": str(e)}, 0.0
    
    def _call_nvidia_api(
        self, 
        image_bytes: bytes, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Call NVIDIA API with Gemma 3-27B-IT to extract concepts.
        
        Args:
            image_bytes: Image bytes
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        logger.warning("NVIDIA API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(image_bytes, prompt)
    
    def _call_openai_api(
        self, 
        image_bytes: bytes, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Call OpenAI's GPT-4 Vision API to extract concepts.
        
        Args:
            image_bytes: Image bytes
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        logger.warning("OpenAI API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(image_bytes, prompt)
    
    def _call_google_api(
        self, 
        image_bytes: bytes, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Call Google's Gemini Vision API to extract concepts.
        
        Args:
            image_bytes: Image bytes
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        logger.warning("Google Gemini API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(image_bytes, prompt)
    
    def _call_anthropic_api(
        self, 
        image_bytes: bytes, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Call Anthropic's Claude Vision API to extract concepts.
        
        Args:
            image_bytes: Image bytes
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        logger.warning("Anthropic Claude API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(image_bytes, prompt)
    
    def _call_local_model(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Use Groq API as fallback instead of rule-based method.
        
        Args:
            image: Image as numpy array
            prompt: Instruction prompt
            
        Returns:
            Tuple of extracted concepts and confidence score
        """
        # Convert image to bytes and call Groq API
        image_bytes = self._convert_image_to_bytes(image)
        return self._call_groq_api(image_bytes, prompt)
    
    def check_grounding(
        self, 
        concepts: Dict[str, Any], 
        image: np.ndarray
    ) -> Dict[str, Any]:
        """Check the visual grounding of concepts using mechanistic interpretability.
        
        Args:
            concepts: Dictionary of concepts from the VLM
            image: Original image
            
        Returns:
            Concepts dictionary with grounding scores
        """
        # If grounding is disabled, return the original concepts with dummy scores
        if not self.enable_grounding:
            logger.info("Grounding check disabled, returning original concepts with placeholder scores")
            print("\n=== GROUNDING DISABLED ===")
            print("Skipping grounding calculation and using default scores")
            print("=========================\n")
            
            # Add placeholder grounding scores
            grounded_concepts = concepts.copy()
            self._add_placeholder_grounding_scores(grounded_concepts)
            return grounded_concepts
        
        # The original grounding implementation follows
        # Create a copy to add grounding scores
        grounded_concepts = concepts.copy()
        
        # Create blank image to compare against (minimal visual context)
        blank_image = np.ones_like(image) * 255
        
        try:
            # Convert both images to bytes for API calls
            image_bytes = self._convert_image_to_bytes(image)
            blank_image_bytes = self._convert_image_to_bytes(blank_image)
            
            # Enhanced base prompt for concept verification with detailed instructions
            base_prompt = """
            Analyze this image thoroughly and determine if the following element is visually present.
            
            When analyzing, please:
            1. Examine the entire image, including peripheral regions
            2. Pay close attention to visual features, objects, and scene elements
            3. Consider context and spatial relationships
            4. Look for subtle indicators and patterns
            5. Assess visibility, clarity, and certainty of identification
            
            Provide your assessment with a confidence level (0-100%).
            """
            
            # Check grounding for scene setting
            if "scene_setting" in grounded_concepts:
                for i, element in enumerate(grounded_concepts["scene_setting"]):
                    try:
                        element_name = element.get("name", "")
                        
                        # Get attributes for more specific verification
                        location = element.get("attributes", {}).get("location", "")
                        appearance = element.get("attributes", {}).get("appearance", "")
                        
                        # Create detailed prompt that includes attributes
                        detail_str = ""
                        if location:
                            detail_str += f" Located in/at: {location}."
                        if appearance:
                            detail_str += f" Appearance characteristics: {appearance}."
                        
                        # Enhanced prompts with attribute-specific instructions
                        image_prompt = f"""
                        {base_prompt}
                        
                        SCENE ELEMENT: '{element_name}'{detail_str}
                        
                        Specific analysis instructions:
                        1. First, identify if '{element_name}' is visually present anywhere in the image
                        2. Then, verify if its location matches '{location}' (if applicable)
                        3. Confirm if its appearance matches '{appearance}' (if applicable)
                        4. Check the overall scene context
                        5. Consider all visual evidence
                        
                        After thorough analysis, is this element visually present in the image? 
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        SCENE ELEMENT: '{element_name}'{detail_str}
                        
                        1. Is '{element_name}' a common setting in emergency rooms or medical facilities?
                        2. Would one expect to find a setting with location '{location}' in a typical scene?
                        3. Is the appearance '{appearance}' a reasonable description without seeing an actual image?
                        
                        Based solely on general knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["scene_setting"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["scene_setting"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for '{element_name}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during scene setting grounding check: {e}")
                        grounded_concepts["scene_setting"][i]["grounding_score"] = 0.5  # Default fallback
            
            # Check grounding for personnel
            if "personnel" in grounded_concepts:
                for i, person in enumerate(grounded_concepts["personnel"]):
                    try:
                        person_name = person.get("name", "")
                        
                        # Get attributes for more specific verification
                        actions = person.get("attributes", {}).get("actions", "")
                        position = person.get("attributes", {}).get("position", "")
                        
                        # Create detailed prompt that includes attributes
                        detail_str = ""
                        if actions:
                            detail_str += f" Performing actions: {actions}."
                        if position:
                            detail_str += f" Position in scene: {position}."
                        
                        # Enhanced prompts with attribute-specific instructions
                        image_prompt = f"""
                        {base_prompt}
                        
                        PERSON: '{person_name}'{detail_str}
                        
                        Specific analysis instructions:
                        1. First, identify if a person with role '{person_name}' is visually present
                        2. Then, verify if they are performing actions like '{actions}' (if applicable)
                        3. Confirm if their position matches '{position}' (if applicable)
                        4. Check their clothing, equipment, and other identifying features
                        5. Consider all contextual clues about their role
                        
                        After thorough analysis, is this person/role visually present in the image? 
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        PERSON: '{person_name}'{detail_str}
                        
                        1. Is '{person_name}' a common role in emergency room or medical settings?
                        2. Would one expect someone in this role to perform actions like '{actions}'?
                        3. Is the position '{position}' a typical location for this role?
                        
                        Based solely on general knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["personnel"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["personnel"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for '{person_name}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during personnel grounding check: {e}")
                        grounded_concepts["personnel"][i]["grounding_score"] = 0.5  # Default fallback
            
            # Check grounding for procedures
            if "procedures" in grounded_concepts:
                for i, procedure in enumerate(grounded_concepts["procedures"]):
                    try:
                        procedure_name = procedure.get("name", "")
                        
                        # Get attributes for more specific verification
                        technique = procedure.get("attributes", {}).get("technique", "")
                        stage = procedure.get("attributes", {}).get("stage", "")
                        
                        # Create detailed prompt that includes attributes
                        detail_str = ""
                        if technique:
                            detail_str += f" Using technique: {technique}."
                        if stage:
                            detail_str += f" Current stage: {stage}."
                        
                        # Enhanced prompts with attribute-specific instructions
                        image_prompt = f"""
                        {base_prompt}
                        
                        PROCEDURE: '{procedure_name}'{detail_str}
                        
                        Specific analysis instructions:
                        1. First, identify if the procedure '{procedure_name}' is being performed
                        2. Look for visual indicators of the specific technique '{technique}' (if applicable)
                        3. Determine if the current stage appears to be '{stage}' (if applicable)
                        4. Check related equipment and hand positions
                        5. Look for patient positioning and provider actions
                        
                        After thorough analysis, is this procedure visually evident in the image? 
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        PROCEDURE: '{procedure_name}'{detail_str}
                        
                        1. Is '{procedure_name}' a common procedure in emergency medical settings?
                        2. Is the technique '{technique}' standard for this procedure?
                        3. Is '{stage}' a typical stage in this procedure?
                        
                        Based solely on medical knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["procedures"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["procedures"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for procedure '{procedure_name}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during procedure grounding check: {e}")
                        grounded_concepts["procedures"][i]["grounding_score"] = 0.5  # Default fallback
            
            # Check grounding for equipment
            if "equipment" in grounded_concepts:
                for i, equipment in enumerate(grounded_concepts["equipment"]):
                    try:
                        equipment_name = equipment.get("name", "")
                        
                        # Get attributes for more specific verification
                        usage = equipment.get("attributes", {}).get("usage", "")
                        position = equipment.get("attributes", {}).get("position", "")
                        
                        # Create detailed prompt that includes attributes
                        detail_str = ""
                        if usage:
                            detail_str += f" Being used for: {usage}."
                        if position:
                            detail_str += f" Positioned at: {position}."
                        
                        # Enhanced prompts with attribute-specific instructions
                        image_prompt = f"""
                        {base_prompt}
                        
                        EQUIPMENT: '{equipment_name}'{detail_str}
                        
                        Specific analysis instructions:
                        1. First, identify if the equipment '{equipment_name}' is visible in the image
                        2. Verify if it appears to be used for '{usage}' (if applicable)
                        3. Check if its position matches '{position}' (if applicable)
                        4. Look for connections to other equipment or people
                        5. Note any distinctive features of this equipment
                        
                        After thorough analysis, is this equipment visually present in the image? 
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        EQUIPMENT: '{equipment_name}'{detail_str}
                        
                        1. Is '{equipment_name}' common equipment in emergency medical settings?
                        2. Is '{usage}' a typical use for this equipment?
                        3. Is '{position}' a standard position for this equipment?
                        
                        Based solely on medical knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["equipment"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["equipment"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for equipment '{equipment_name}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during equipment grounding check: {e}")
                        grounded_concepts["equipment"][i]["grounding_score"] = 0.5  # Default fallback
            
            # Check grounding for clinical elements
            if "clinical_elements" in grounded_concepts:
                for i, element in enumerate(grounded_concepts["clinical_elements"]):
                    try:
                        element_name = element.get("name", "")
                        
                        # Get attributes for more specific verification
                        significance = element.get("attributes", {}).get("significance", "")
                        appearance = element.get("attributes", {}).get("appearance", "")
                        
                        # Create detailed prompt that includes attributes
                        detail_str = ""
                        if significance:
                            detail_str += f" Clinical significance: {significance}."
                        if appearance:
                            detail_str += f" Appearance: {appearance}."
                        
                        # Enhanced prompts with attribute-specific instructions
                        image_prompt = f"""
                        {base_prompt}
                        
                        CLINICAL ELEMENT: '{element_name}'{detail_str}
                        
                        Specific analysis instructions:
                        1. First, identify if the clinical element '{element_name}' is visible
                        2. Consider its apparent clinical significance in line with '{significance}' (if applicable)
                        3. Verify if its appearance matches '{appearance}' (if applicable)
                        4. Look for related indicators or equipment
                        5. Consider the patient's condition if visible
                        
                        After thorough analysis, is this clinical element visually evident in the image? 
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        CLINICAL ELEMENT: '{element_name}'{detail_str}
                        
                        1. Is '{element_name}' a common clinical element in emergency settings?
                        2. Would this element typically have significance like '{significance}'?
                        3. Is the appearance '{appearance}' a typical presentation?
                        
                        Based solely on medical knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["clinical_elements"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["clinical_elements"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for clinical element '{element_name}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during clinical element grounding check: {e}")
                        grounded_concepts["clinical_elements"][i]["grounding_score"] = 0.5  # Default fallback
            
            # Check grounding for relationships - this part remains largely the same
            if "relationships" in grounded_concepts:
                for i, relationship in enumerate(grounded_concepts["relationships"]):
                    try:
                        # Safely access relationship fields with fallbacks
                        source = relationship.get('source', relationship.get('subject', ''))
                        relation = relationship.get('relation', '')
                        target = relationship.get('target', relationship.get('object', ''))
                        
                        rel_desc = f"{source} {relation} {target}"
                        
                        # Enhanced prompts with specific spatial relationship analysis
                        image_prompt = f"""
                        {base_prompt}
                        
                        RELATIONSHIP: '{rel_desc}'
                        
                        Specific analysis instructions:
                        1. First, identify if both '{source}' and '{target}' are visible in the image
                        2. Carefully examine the spatial relationship between them
                        3. Determine if their arrangement matches the relationship '{relation}'
                        4. Consider the context and typical emergency room arrangements
                        5. Look at surrounding objects or people for contextual evidence
                        
                        Is there visual evidence in the image supporting this specific relationship?
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        blank_prompt = f"""
                        Without any visual evidence and looking only at a blank image, consider:
                        
                        RELATIONSHIP: '{rel_desc}'
                        
                        1. Is '{source}' typically related to '{target}' in emergency medical contexts?
                        2. Is '{relation}' a common relationship between these two entities?
                        3. Would this relationship be expected in a typical emergency room scene?
                        
                        Based solely on general knowledge (not visual evidence), provide your assessment.
                        Respond with YES or NO, followed by your confidence level (0-100%).
                        """
                        
                        # Get responses for both prompts
                        image_response, _ = self._call_groq_api(image_bytes, image_prompt)
                        blank_response, _ = self._call_groq_api(blank_image_bytes, blank_prompt)
                        
                        # Calculate grounding score based on response difference
                        image_confidence = self._extract_confidence(image_response)
                        blank_confidence = self._extract_confidence(blank_response)
                        grounding_score = max(0, image_confidence - blank_confidence)
                        
                        # Store the raw responses for debugging
                        grounded_concepts["relationships"][i]["grounding_details"] = {
                            "image_prompt_response": str(image_response).strip(),
                            "blank_prompt_response": str(blank_response).strip(),
                            "image_confidence": image_confidence,
                            "blank_confidence": blank_confidence
                        }
                        
                        grounded_concepts["relationships"][i]["grounding_score"] = grounding_score
                        
                        print(f"Grounding check for relationship '{rel_desc}': score={grounding_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error during relationship grounding check: {e}")
                        grounded_concepts["relationships"][i]["grounding_score"] = 0.5  # Default fallback
        
        except Exception as e:
            logger.error(f"Error in grounding check: {e}")
            # Fall back to simple random grounding scores
            self._fallback_grounding_scores(grounded_concepts)
        
        # Print the grounding scores
        print("\n=== GROUNDING SCORES ===")
        print(json.dumps(grounded_concepts, indent=2))
        print("=======================\n")
        
        return grounded_concepts
    
    def _add_placeholder_grounding_scores(self, concepts: Dict[str, Any]) -> None:
        """Add placeholder grounding scores when grounding is disabled.
        
        Args:
            concepts: Concepts dictionary to modify in-place
        """
        # Add placeholder grounding scores 
        if "scene_setting" in concepts:
            for i, setting in enumerate(concepts["scene_setting"]):
                concepts["scene_setting"][i]["grounding_score"] = 0.85
        
        if "personnel" in concepts:
            for i, person in enumerate(concepts["personnel"]):
                concepts["personnel"][i]["grounding_score"] = 0.85
        
        if "procedures" in concepts:
            for i, procedure in enumerate(concepts["procedures"]):
                concepts["procedures"][i]["grounding_score"] = 0.8
        
        if "equipment" in concepts:
            for i, equipment in enumerate(concepts["equipment"]):
                concepts["equipment"][i]["grounding_score"] = 0.85
        
        if "clinical_elements" in concepts:
            for i, element in enumerate(concepts["clinical_elements"]):
                concepts["clinical_elements"][i]["grounding_score"] = 0.8
        
        if "relationships" in concepts:
            for i, relationship in enumerate(concepts["relationships"]):
                concepts["relationships"][i]["grounding_score"] = 0.75
    
    def _fallback_grounding_scores(self, concepts: Dict[str, Any]) -> None:
        """Fallback method to assign random grounding scores when the main method fails.
        
        Args:
            concepts: Concepts dictionary to modify in-place
        """
        # Add random grounding scores as placeholder
        if "scene_setting" in concepts:
            for i, setting in enumerate(concepts["scene_setting"]):
                concepts["scene_setting"][i]["grounding_score"] = np.random.uniform(0.7, 0.95)
        
        if "personnel" in concepts:
            for i, person in enumerate(concepts["personnel"]):
                concepts["personnel"][i]["grounding_score"] = np.random.uniform(0.7, 0.95)
        
        if "procedures" in concepts:
            for i, procedure in enumerate(concepts["procedures"]):
                concepts["procedures"][i]["grounding_score"] = np.random.uniform(0.6, 0.9)
        
        if "equipment" in concepts:
            for i, equipment in enumerate(concepts["equipment"]):
                concepts["equipment"][i]["grounding_score"] = np.random.uniform(0.7, 0.95)
        
        if "clinical_elements" in concepts:
            for i, element in enumerate(concepts["clinical_elements"]):
                concepts["clinical_elements"][i]["grounding_score"] = np.random.uniform(0.6, 0.9)
        
        if "relationships" in concepts:
            for i, relationship in enumerate(concepts["relationships"]):
                concepts["relationships"][i]["grounding_score"] = np.random.uniform(0.5, 0.85)
        
        logger.warning("Used fallback random grounding scores due to API error")
    
    def _extract_confidence(self, response: Dict[str, Any]) -> float:
        """Extract confidence score from yes/no response with percentage.
        
        Args:
            response: Response from the VLM
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Handle different response types
            if isinstance(response, dict) and "raw_response" in response:
                response_text = str(response["raw_response"]).lower()
            else:
                response_text = str(response).lower()
            
            # Check for YES/NO
            is_yes = any(marker in response_text for marker in ["yes", "present", "visible", "evident", "confirmed"])
            is_no = any(marker in response_text for marker in ["no", "not present", "not visible", "cannot see", "absent"])
            
            # Look for confidence percentage
            percentage_matches = re.findall(r'(\d{1,3})(?:\s*%|\s*percent)', response_text)
            confidence_matches = re.findall(r'confidence[:\s]+(\d+(?:\.\d+)?)', response_text)
            
            # Extract numeric confidence value
            if percentage_matches:
                # Use percentage from response
                confidence_value = float(percentage_matches[0]) / 100.0
            elif confidence_matches:
                # Use confidence value from response
                confidence_value = float(confidence_matches[0])
                # If it's on a 0-100 scale, normalize to 0-1
                if confidence_value > 1.0:
                    confidence_value /= 100.0
            else:
                # Default confidence values based on yes/no
                confidence_value = 0.9 if is_yes else 0.1
            
            # Adjust confidence based on certainty language
            if "uncertain" in response_text or "unclear" in response_text or "difficult to determine" in response_text:
                confidence_value *= 0.7
            if "definitely" in response_text or "clearly" in response_text or "certain" in response_text:
                confidence_value = min(0.95, confidence_value * 1.2)
            
            return confidence_value
                
        except Exception as e:
            logger.error(f"Error extracting confidence: {e}")
            return 0.5  # Default neutral confidence