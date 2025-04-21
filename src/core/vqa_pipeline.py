"""Main Medical VQA pipeline integrating all components."""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import numpy as np

from src.utils.image_utils import load_image, apply_roi_mask, save_visualization
from src.utils.nlp_utils import extract_key_concepts, identify_question_type, get_traversal_strategy
from src.core.roi_detector import ROIDetector
from src.core.concept_extractor import ConceptExtractor
from src.graphs.knowledge_graph import MedicalKnowledgeGraph
from src.core.answer_synthesizer import AnswerSynthesizer
from src.config import ROI_DETECTION_ENABLED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalVQAPipeline:
    """Main pipeline for Medical VQA with hallucination mitigation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        roi_model: Optional[str] = None,
        enable_grounding: bool = True,
        enable_roi_detection: Optional[bool] = None
    ):
        """Initialize the Medical VQA pipeline.
        
        Args:
            api_key: API key for VLM services
            model: Model identifier for the VLM
            roi_model: Model for ROI detection
            enable_grounding: Whether to perform grounding score calculation (default: True)
            enable_roi_detection: Whether to use ROI detection or use the entire image (defaults to config value)
        """
        # Initialize components
        roi_model_name = roi_model 
        
        # Set ROI detection flag, using config default if not specified
        self.enable_roi_detection = ROI_DETECTION_ENABLED if enable_roi_detection is None else enable_roi_detection
        
        logger.info(f"Initializing Medical VQA Pipeline with model: {model}")
        print(f"\n=== PIPELINE INITIALIZATION ===")
        print(f"VLM Model: {model}")
        print(f"ROI Model: {roi_model_name}")
        print(f"Grounding Enabled: {enable_grounding}")
        print(f"ROI Detection Enabled: {self.enable_roi_detection}")
        print("===============================\n")
        
        self.roi_detector = ROIDetector(model_name=roi_model_name)
        self.concept_extractor = ConceptExtractor(api_key=api_key, model=model, enable_grounding=enable_grounding)
        self.knowledge_graph = MedicalKnowledgeGraph()
        self.answer_synthesizer = AnswerSynthesizer(api_key=api_key, model=model)
        
        # Track execution times for analysis
        self.execution_times = {}
    
    def answer_question(
        self, 
        image_path: str, 
        question: str,
        output_dir: Optional[str] = None,
        enable_roi_detection: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Process an image and question to produce an answer with hallucination mitigation.
        
        Args:
            image_path: Path to the medical image
            question: Medical question about the image
            output_dir: Optional directory to save results and visualizations
            enable_roi_detection: Override the default ROI detection setting for this question
            
        Returns:
            Dictionary with answer and additional information
        """
        start_time = time.time()
        logger.info(f"Processing question: {question}")
        print(f"\n========== STARTING NEW QUERY ==========")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Output Directory: {output_dir or 'Not specified'}")
        print("=========================================\n")
        
        # Use instance default for ROI detection if not specified for this question
        use_roi = self.enable_roi_detection if enable_roi_detection is None else enable_roi_detection
        
        # Track interim outputs for debugging and analysis
        interim_results = {}
        
        # Phase 1: Question Analysis
        t0 = time.time()
        print("\n=== PHASE 1: QUESTION ANALYSIS ===")
        medical_concepts = extract_key_concepts(question)
        question_type = identify_question_type(question)
        traversal_strategy = get_traversal_strategy(question_type)
        
        print(f"Extracted Medical Concepts: {medical_concepts}")
        print(f"Question Type: {question_type}")
        print(f"Selected Traversal Strategy: {traversal_strategy}")
        print(f"ROI Detection Enabled: {use_roi}")
        print("===================================\n")
        
        interim_results["question_analysis"] = {
            "concepts": medical_concepts,
            "question_type": question_type,
            "traversal_strategy": traversal_strategy,
            "roi_detection_enabled": use_roi
        }
        self.execution_times["question_analysis"] = time.time() - t0
        logger.info(f"Question analysis complete. Type: {question_type}")
        
        # Load original image 
        original_image = load_image(image_path)
        print(f"Original Image Shape: {original_image.shape}")
        
        # If ROI detection is enabled, proceed with detection and masking
        focused_image = original_image  # Default to using the whole image
        roi_coords = []
        
        if use_roi:
            # Phase 2: ROI Detection (if enabled)
            t0 = time.time()
            print("\n=== PHASE 2: ROI DETECTION ===")
            roi_coords = self.roi_detector.detect(image_path, question)
            print(f"Detected ROI Coordinates: {roi_coords}")
            
            # Apply focus mask if ROIs found
            if roi_coords:
                focused_image = apply_roi_mask(original_image, roi_coords)
                logger.info(f"Found {len(roi_coords)} regions of interest")
                print(f"Applied mask to {len(roi_coords)} regions of interest")
            else:
                # If no ROI found, use the whole image
                h, w = original_image.shape[:2]
                roi_coords = [(0, 0, w, h)]
                logger.info("No specific ROI found, using entire image")
                print("No specific ROI found, using entire image")
            
            # Save focused image for inspection
            if output_dir:
                focused_img_path = os.path.join(output_dir, f"focused_{os.path.basename(image_path)}")
                save_visualization(focused_image, focused_img_path, question)
                print(f"Saved focused image to: {focused_img_path}")
            
            print(f"Focused Image Shape: {focused_image.shape}")
            print("=============================\n")
            
            interim_results["roi_detection"] = {
                "roi_coords": roi_coords,
                "roi_detection_enabled": use_roi
            }
            self.execution_times["roi_detection"] = time.time() - t0
        else:
            # Skip ROI detection, using whole image
            print("\n=== PHASE 2: SKIPPED ROI DETECTION ===")
            print("ROI detection disabled, using the entire image")
            h, w = original_image.shape[:2]
            roi_coords = [(0, 0, w, h)]
            print(f"Using entire image with dimensions: {w}x{h}")
            print("======================================\n")
            
            interim_results["roi_detection"] = {
                "roi_coords": roi_coords,
                "roi_detection_enabled": use_roi
            }
            self.execution_times["roi_detection"] = 0  # No time spent on ROI detection
        
        # Phase 3: Concept Extraction with VLM
        t0 = time.time()
        print("\n=== PHASE 3: CONCEPT EXTRACTION ===")
        print("Extracting concepts from image using VLM...")
        concepts, confidence = self.concept_extractor.extract_concepts(focused_image)
        
        print(f"Extraction Confidence: {confidence:.4f}")
        print("==================================\n")
        
        interim_results["concept_extraction"] = {
            "raw_concepts": concepts,
            "extraction_confidence": confidence
        }
        self.execution_times["concept_extraction"] = time.time() - t0
        logger.info(f"Extracted concepts with confidence: {confidence:.2f}")
        
        # Phase 4: Grounding Check with MI
        t0 = time.time()
        print("\n=== PHASE 4: GROUNDING CHECK ===")
        print("Verifying visual grounding of extracted concepts...")
        grounded_concepts = self.concept_extractor.check_grounding(concepts, focused_image)
        
        print("===============================\n")
        
        interim_results["grounding_check"] = {
            "grounded_concepts": grounded_concepts
        }
        self.execution_times["grounding_check"] = time.time() - t0
        
        # Phase 5: Knowledge Graph Construction
        t0 = time.time()
        print("\n=== PHASE 5: KNOWLEDGE GRAPH CONSTRUCTION ===")
        image_id = os.path.basename(image_path)
        print(f"Building knowledge graph for image ID: {image_id}")
        self.knowledge_graph.build_from_concepts(grounded_concepts, image_id)
        
        print(f"Knowledge Graph Stats:")
        print(f"- Nodes: {self.knowledge_graph.graph.number_of_nodes()}")
        print(f"- Edges: {self.knowledge_graph.graph.number_of_edges()}")
        print(f"- Node Types: {set([data.get('type', 'unknown') for _, data in self.knowledge_graph.graph.nodes(data=True)])}")
        
        # Print knowledge graph structure
        print("\nKnowledge Graph Structure:")
        for node, data in self.knowledge_graph.graph.nodes(data=True):
            print(f"Node: {node} (Type: {data.get('type', 'unknown')})")
            for attr, value in data.items():
                if attr != 'type':
                    print(f"  {attr}: {value}")
        
        print("\nKnowledge Graph Edges:")
        for u, v, data in self.knowledge_graph.graph.edges(data=True):
            print(f"{u} --[{data.get('relation', 'related_to')}]--> {v}")
        
        # Save graph visualization if output directory is provided
        if output_dir:
            graph_vis_path = os.path.join(output_dir, f"graph_{os.path.basename(image_path)}.png")
            self.knowledge_graph.visualize(graph_vis_path)
            print(f"Graph visualization saved to: {graph_vis_path}")
            
            # Save graph data as JSON
            graph_data_path = os.path.join(output_dir, f"graph_{os.path.basename(image_path)}.json")
            with open(graph_data_path, 'w') as f:
                graph_json = self.knowledge_graph.to_json()
                f.write(graph_json)
                print(f"Graph data saved to: {graph_data_path}")
                print("\nGraph JSON Representation:")
                print(graph_json)
        
        print("===========================================\n")
        
        self.execution_times["graph_construction"] = time.time() - t0
        logger.info(f"Knowledge graph built with {self.knowledge_graph.graph.number_of_nodes()} nodes")
        
        # Phase 6: Graph Traversal
        t0 = time.time()
        print("\n=== PHASE 6: GRAPH TRAVERSAL ===")
        print(f"Traversing graph using strategy: {traversal_strategy}")
        
        # Extract a caption from the image if available
        image_caption = self._extract_image_caption(focused_image)
        print(f"Image Caption: {image_caption or 'None'}")
        
        # Pass image caption to traversal
        graph_result = self.knowledge_graph.traverse(question, traversal_strategy, image_caption)
        
        print("Graph Traversal Result:")
        print(json.dumps(graph_result, indent=2))
        print("==============================\n")
        
        interim_results["graph_traversal"] = {
            "query_result": graph_result,
            "traversal_strategy": traversal_strategy,
            "image_caption": image_caption
        }
        self.execution_times["graph_traversal"] = time.time() - t0
        
        # Phase 7: Answer Synthesis
        t0 = time.time()
        print("\n=== PHASE 7: ANSWER SYNTHESIS ===")
        print("Synthesizing final answer...")
        answer = self.answer_synthesizer.synthesize_answer(
            question, 
            graph_result,
            image_caption=image_caption
        )
        
        print(f"\nFinal Answer: {answer}")
        print("===============================\n")
        
        self.execution_times["answer_synthesis"] = time.time() - t0
        logger.info("Answer synthesis complete")
        
        # Calculate total execution time
        total_time = time.time() - start_time
        self.execution_times["total"] = total_time
        
        # Execution Time Summary
        print("\n=== EXECUTION TIME SUMMARY ===")
        for phase, duration in self.execution_times.items():
            print(f"{phase}: {duration:.2f} seconds")
        print(f"Total execution time: {total_time:.2f} seconds")
        print("=============================\n")
        
        # Prepare final result
        result = {
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "roi_detection_enabled": use_roi,
            "execution_times": self.execution_times,
            "interim_results": interim_results
        }
        
        # Save result to file if output directory is provided
        if output_dir:
            result_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}.json")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Full results saved to: {result_path}")
        
        print("\n========== END OF QUERY ==========\n")
        
        logger.info(f"Question answered in {total_time:.2f} seconds")
        return result
        
    def _extract_image_caption(self, image: np.ndarray) -> Optional[str]:
        """Extract a basic caption from the image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Caption string or None if extraction failed
        """
        try:
            # Use concept extractor to get a simplified caption
            # Here we're just extracting the main scene elements and combining them
            concepts, _ = self.concept_extractor.extract_concepts(image)
            
            caption_parts = []
            
            # Add scene setting
            if "scene_setting" in concepts and concepts["scene_setting"]:
                scene_parts = [item["name"] for item in concepts["scene_setting"]]
                if scene_parts:
                    caption_parts.append(f"Setting: {', '.join(scene_parts)}")
            
            # Add personnel
            if "personnel" in concepts and concepts["personnel"]:
                personnel_parts = [item["name"] for item in concepts["personnel"]]
                if personnel_parts:
                    caption_parts.append(f"People: {', '.join(personnel_parts)}")
            
            # Add procedures
            if "procedures" in concepts and concepts["procedures"]:
                procedure_parts = [item["name"] for item in concepts["procedures"]]
                if procedure_parts:
                    caption_parts.append(f"Activities: {', '.join(procedure_parts)}")
            
            # Add equipment
            if "equipment" in concepts and concepts["equipment"]:
                equipment_parts = [item["name"] for item in concepts["equipment"]]
                if equipment_parts:
                    caption_parts.append(f"Equipment: {', '.join(equipment_parts)}")
            
            # Combine all parts
            if caption_parts:
                return ". ".join(caption_parts)
            return None
        except Exception as e:
            logger.error(f"Error extracting image caption: {e}")
            return None 