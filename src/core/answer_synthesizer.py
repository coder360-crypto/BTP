"""Answer synthesizer for Medical VQA."""

import json
import requests
import logging
import base64
from typing import Dict, Any, Optional, List
import os
from groq import Groq

from src.config import VLM_API_KEY, VLM_MODEL, VLM_TEMP, ANSWER_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """Synthesizes final answers for medical VQA based on knowledge graph context."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the answer synthesizer.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier for the LLM
        """
        self.api_key =  "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"
        self.model =  "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Check if required credentials are available
        if not self.api_key:
            logger.warning("API key not provided. Set VLM_API_KEY in .env or provide in constructor.")
        
        # Configure for provider
        self.provider = "groq"
    
    def _determine_provider(self) -> str:
        """Determine which LLM provider to use based on model name."""
        # Always use Groq API
        return "groq"
    
    def synthesize_answer(
        self, 
        question: str, 
        graph_result: Dict[str, Any],
        custom_prompt: Optional[str] = None,
        image_caption: Optional[str] = None
    ) -> str:
        """Synthesize a final answer based on the question and graph traversal results.
        
        Args:
            question: The medical question
            graph_result: Results from knowledge graph traversal
            custom_prompt: Optional custom prompt template
            image_caption: Optional caption extracted from the image
            
        Returns:
            Final answer text
        """
        # Check if this is a descriptive query with empty graph
        from src.utils.nlp_utils import identify_question_type
        question_type = identify_question_type(question)
        is_descriptive = question_type == 'descriptive'
        
        # Check if graph is empty (no nodes or edges)
        is_empty_graph = False
        if "nodes" in graph_result and "edges" in graph_result:
            is_empty_graph = len(graph_result.get("nodes", [])) == 0 or len(graph_result.get("edges", [])) == 0
        
        # For descriptive queries with empty graphs, rely on image caption if available
        if is_descriptive and is_empty_graph and image_caption:
            prompt_template = custom_prompt or ANSWER_SYNTHESIS_PROMPT
            
            # Create special context using only the image caption
            context = f"Image caption: {image_caption}\n\nNo detailed graph information is available, but the image shows: {image_caption}"
            
            # Format the full prompt
            prompt = prompt_template.format(
                question=question,
                context=context
            )
            
            # Call the Groq API
            return self._call_groq_api(prompt)
        
        # Format graph context for the LLM
        graph_context = self._format_graph_context(graph_result)
        
        # Add image caption to context if available and not already included
        if image_caption and "IMAGE CAPTION:" not in graph_context:
            graph_context = f"IMAGE CAPTION: {image_caption}\n\n{graph_context}"
        
        # Print the knowledge graph
        print("\n=== KNOWLEDGE GRAPH ===")
        print(json.dumps(graph_result, indent=2))
        print("======================\n")
        
        # Get the appropriate prompt template
        prompt_template = custom_prompt or ANSWER_SYNTHESIS_PROMPT
        
        # Format the full prompt
        prompt = prompt_template.format(
            question=question,
            context=graph_context
        )
        
        # Call the Groq API
        return self._call_groq_api(prompt)
    
    def _format_graph_context(self, graph_result: Dict[str, Any]) -> str:
        """Format the graph traversal results into a text context for the LLM.
        
        Args:
            graph_result: Results from knowledge graph traversal
            
        Returns:
            Formatted context string
        """
        # Check if this result is caption-based (for empty graphs)
        if graph_result.get("caption_based") and graph_result.get("image_caption"):
            return f"Based on the image caption: {graph_result['image_caption']}\n\nNo detailed medical concepts were identified in the image."
            
        context_parts = []
        
        # Include image caption if available
        if graph_result.get("image_caption"):
            context_parts.append(f"IMAGE CAPTION: {graph_result['image_caption']}\n")
        
        # Check if this is a descriptive scene analysis result
        if "scene_found" in graph_result and "scene_summary" in graph_result:
            return self._format_scene_analysis(graph_result)
            
        # Check if this is a procedure evaluation result
        if "evaluation_possible" in graph_result and "procedures" in graph_result and "evaluation_focus" in graph_result:
            return self._format_procedure_evaluation(graph_result)
            
        # Check if this is a reasoning traversal result
        if "reasoning_applied" in graph_result and "graph_structure" in graph_result:
            return self._format_reasoning_traversal(graph_result)
        
        # Include different sections based on traversal strategy used
        if "exists" in graph_result:
            # Node existence results
            if graph_result["exists"]:
                context_parts.append("The following concepts were identified in the image:")
                for node in graph_result.get("matching_nodes", []):
                    context_parts.append(f"- {node['name']} ({node['type']})")
                    # Include attributes if any
                    if node.get("attributes"):
                        for attr, value in node["attributes"].items():
                            context_parts.append(f"  - {attr}: {value}")
            else:
                context_parts.append("The requested concept was not found in the image.")
        
        elif "found" in graph_result and "attributes" in graph_result:
            # Attribute retrieval results
            if graph_result["found"]:
                context_parts.append("Found the following information:")
                
                # Include attributes
                for attr_group in graph_result.get("attributes", []):
                    context_parts.append(f"- {attr_group['node']} ({attr_group['type']}):")
                    for prop, value in attr_group.get("properties", {}).items():
                        context_parts.append(f"  - {prop}: {value}")
                
                # Include relationships
                if graph_result.get("relationships"):
                    context_parts.append("\nRelationships:")
                    for rel in graph_result["relationships"]:
                        context_parts.append(f"- {rel['from']} {rel['relation']} {rel['to']}")
            else:
                context_parts.append("The requested information was not found in the image.")
        
        elif "paths_found" in graph_result:
            # Path finding results
            if graph_result["paths_found"]:
                context_parts.append("Found the following relationships:")
                
                for path_idx, path in enumerate(graph_result.get("paths", [])):
                    context_parts.append(f"\nPath {path_idx + 1}:")
                    
                    # Format relation path
                    for rel in path.get("relations", []):
                        context_parts.append(f"- {rel['from']} {rel['relation']} {rel['to']}")
            else:
                context_parts.append("No relationships were found between the specified concepts.")
        
        elif "comparison_possible" in graph_result:
            # Comparison results
            if graph_result["comparison_possible"]:
                entities = graph_result.get("entities", [])
                entity_names = [entity.get("name", "") for entity in entities]
                
                context_parts.append(f"Comparison between {' and '.join(entity_names)}:")
                
                # Common attributes
                if graph_result.get("common_attributes"):
                    context_parts.append("\nShared attributes:")
                    for attr in graph_result["common_attributes"]:
                        values_str = " vs ".join([str(v) for v in attr["values"]])
                        context_parts.append(f"- {attr['attribute']}: {values_str}")
                
                # Differences
                if graph_result.get("differences"):
                    context_parts.append("\nDifferences:")
                    for diff in graph_result["differences"]:
                        values_with_entities = []
                        for i, value in enumerate(diff["values"]):
                            if i < len(entity_names):
                                values_with_entities.append(f"{entity_names[i]}: {value}")
                        context_parts.append(f"- {diff['attribute']}: {', '.join(values_with_entities)}")
            else:
                context_parts.append("Unable to compare the requested concepts.")
        
        elif "nodes" in graph_result and "edges" in graph_result:
            # Check if we have any nodes or edges
            if not graph_result.get("nodes") and not graph_result.get("edges"):
                context_parts.append("No specific medical concepts or relationships were identified in the image.")
            else:
                # Subgraph extraction results
                if graph_result.get("found", True):  # Default to True if "found" is not present
                    # Nodes
                    context_parts.append("Identified components:")
                    for node in graph_result.get("nodes", []):
                        context_parts.append(f"- {node['name']} ({node['type']})")
                        # Include attributes if any
                        if node.get("attributes"):
                            for attr, value in node["attributes"].items():
                                context_parts.append(f"  - {attr}: {value}")
                    
                    # Relationships
                    if graph_result.get("edges"):
                        context_parts.append("\nRelationships:")
                        for edge in graph_result["edges"]:
                            # Get source and target names from node IDs
                            source_id = edge["source"]
                            target_id = edge["target"]
                            
                            # Find corresponding node entries
                            source_name = source_id
                            target_name = target_id
                            
                            for node in graph_result.get("nodes", []):
                                if node["id"] == source_id:
                                    source_name = node["name"]
                                if node["id"] == target_id:
                                    target_name = node["name"]
                            
                            context_parts.append(f"- {source_name} {edge['relation']} {target_name}")
                else:
                    context_parts.append("The requested information was not found in the image.")
        
        # Join all context parts with newlines
        return "\n".join(context_parts)
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API with Meta's Llama model to generate an answer.
        
        Args:
            prompt: Full prompt for the LLM
            
        Returns:
            Generated answer text
        """
        try:
            # Setup Groq client
            client = Groq(api_key=self.api_key)
            
            # Make API call
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=VLM_TEMP,
                max_completion_tokens=500,
                top_p=0.7,
                stream=False
            )
            
            # Extract content from Groq API response
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _call_nvidia_api(self, prompt: str) -> str:
        """Call NVIDIA API with Gemma 3-27B-IT to generate an answer.
        
        Args:
            prompt: Full prompt for the LLM
            
        Returns:
            Generated answer text
        """
        logger.warning("NVIDIA API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(prompt)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI's API to generate an answer.
        
        Args:
            prompt: Full prompt for the LLM
            
        Returns:
            Generated answer text
        """
        logger.warning("OpenAI API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(prompt)
    
    def _call_google_api(self, prompt: str) -> str:
        """Call Google's API to generate an answer.
        
        Args:
            prompt: Full prompt for the LLM
            
        Returns:
            Generated answer text
        """
        logger.warning("Google API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(prompt)
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic's API to generate an answer.
        
        Args:
            prompt: Full prompt for the LLM
            
        Returns:
            Generated answer text
        """
        logger.warning("Anthropic API method is deprecated. Using Groq API instead.")
        return self._call_groq_api(prompt)
    
    def _generate_simple_answer(
        self, 
        question: str, 
        graph_result: Dict[str, Any]
    ) -> str:
        """Generate a simple answer based on the graph result using Groq API.
        
        Args:
            question: The medical question
            graph_result: Results from knowledge graph traversal
            
        Returns:
            Simple answer text
        """
        # Format graph context
        graph_context = self._format_graph_context(graph_result)
        
        # Create a simple prompt for the Groq API
        prompt = f"""
        Based on the following information extracted from a medical image, please provide a brief answer to this question: {question}
        
        Information from image:
        {graph_context}
        
        Please provide a brief and clinically appropriate response:
        """
        
        # Use Groq API instead of rule-based approach
        return self._call_groq_api(prompt)
    
    def _format_procedure_evaluation(self, eval_result: Dict[str, Any]) -> str:
        """Format the procedure evaluation results into a structured context for the LLM.
        
        Args:
            eval_result: Results from procedure evaluation
            
        Returns:
            Formatted procedure evaluation context
        """
        context_parts = []
        
        # Check if evaluation was possible
        if not eval_result["evaluation_possible"]:
            return "No clear procedures or relevant information were identified in the image."
        
        # Check if procedures were identified
        if not eval_result["procedure_identified"]:
            return "No specific medical procedures were identified in the image."
        
        # Get evaluation focus
        eval_focus = eval_result.get("evaluation_focus", "general_correctness")
        
        # Add procedures section
        context_parts.append("MEDICAL PROCEDURES IDENTIFIED:")
        for proc in eval_result["procedures"]:
            context_parts.append(f"- {proc['name']}")
            # Add detailed attributes based on the focus
            for attr, value in proc["attributes"].items():
                context_parts.append(f"  - {attr}: {value}")
        
        # Add personnel involved
        if eval_result["personnel_involved"]:
            context_parts.append("\nPERSONNEL PERFORMING PROCEDURES:")
            for person in eval_result["personnel_involved"]:
                context_parts.append(f"- {person['name']}")
                # Add attributes relevant to safety/technique based on focus
                if eval_focus in ["technique", "safety", "ppe"]:
                    for attr, value in person["attributes"].items():
                        if any(term in attr.lower() for term in ["wearing", "using", "ppe", "protection", "technique", "training"]):
                            context_parts.append(f"  - {attr}: {value}")
        
        # Add equipment section with focus on evaluation
        if eval_result["related_equipment"]:
            context_parts.append("\nRELATED MEDICAL EQUIPMENT:")
            for equip in eval_result["related_equipment"]:
                context_parts.append(f"- {equip['name']}")
                # Add attributes relevant to the evaluation focus
                if eval_focus in ["equipment_usage", "technique", "safety"]:
                    for attr, value in equip["attributes"].items():
                        context_parts.append(f"  - {attr}: {value}")
        
        # Add key relationships focused on evaluation
        if eval_result["key_relationships"]:
            context_parts.append("\nPROCEDURE RELATIONSHIPS:")
            
            # Filter relationships based on focus
            focus_rels = []
            for rel in eval_result["key_relationships"]:
                # For technique focus, prioritize relationships about how procedures are performed
                if eval_focus == "technique" and rel["relation"] in ["performed_with", "using", "requires", "technique_is"]:
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
                
                # For safety focus, prioritize relationships about safety measures
                elif eval_focus == "safety" and any(term in rel["relation"] for term in ["safety", "protection", "prevents", "protocol"]):
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
                
                # For PPE focus, prioritize relationships about protective equipment
                elif eval_focus == "ppe" and any(term in rel["source_name"].lower() or term in rel["target_name"].lower() 
                                               for term in ["ppe", "mask", "glove", "gown", "shield", "protection"]):
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
                
                # For equipment usage focus, prioritize relationships about equipment
                elif eval_focus == "equipment_usage" and any(term in rel["relation"] for term in ["uses", "requires", "utilizing"]):
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
                
                # For procedure steps focus, prioritize relationships about steps/sequence
                elif eval_focus == "procedure_steps" and any(term in rel["relation"] for term in ["followed_by", "precedes", "step", "sequence"]):
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
                
                # Include all relationships if no focus-specific ones or for general evaluation
                elif not focus_rels or eval_focus == "general_correctness":
                    focus_rels.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
            
            # Add the filtered relationships
            for rel_str in focus_rels:
                context_parts.append(rel_str)
        
        # Add evaluation focus information
        context_parts.append(f"\nEVALUATION FOCUS: {eval_focus.replace('_', ' ').title()}")
        
        # Join all context parts with newlines
        return "\n".join(context_parts)
        
    def _add_scene_elements(self, context_parts: List[str], scene_result: Dict[str, Any]) -> None:
        """Add scene elements to the context parts list.
        
        Args:
            context_parts: List of context parts to append to
            scene_result: Scene analysis result dictionary
        """
        # Setting/Environment
        if scene_result["setting"]:
            context_parts.append("Setting/Environment:")
            for setting in scene_result["setting"]:
                context_parts.append(f"- {setting['name']}")
        
        # Personnel
        if scene_result["personnel"]:
            context_parts.append("\nHealthcare Personnel:")
            for person in scene_result["personnel"]:
                context_parts.append(f"- {person['name']}")
        
        # Procedures
        if scene_result["procedures"]:
            context_parts.append("\nMedical Procedures:")
            for proc in scene_result["procedures"]:
                context_parts.append(f"- {proc['name']}")
        
        # Equipment
        if scene_result["equipment"]:
            context_parts.append("\nMedical Equipment:")
            for equip in scene_result["equipment"]:
                context_parts.append(f"- {equip['name']}")
        
        # Clinical Findings
        if scene_result["clinical_findings"]:
            context_parts.append("\nClinical Findings:")
            for finding in scene_result["clinical_findings"]:
                context_parts.append(f"- {finding['name']}")
    
    def _add_relationships(self, context_parts: List[str], scene_result: Dict[str, Any]) -> None:
        """Add key relationships to the context parts list.
        
        Args:
            context_parts: List of context parts to append to
            scene_result: Scene analysis result dictionary
        """
        if scene_result["relationships"]:
            context_parts.append("\nKey Relationships:")
            added_rels = set()  # Track added relationships to avoid duplicates
            
            for rel in scene_result["relationships"]:
                rel_str = f"{rel['source_name']} {rel['relation']} {rel['target_name']}"
                if rel_str not in added_rels:
                    context_parts.append(f"- {rel_str}")
                    added_rels.add(rel_str) 
    
    def _format_scene_analysis(self, scene_result: Dict[str, Any]) -> str:
        """Format the scene analysis results into a structured context for the LLM.
        
        Args:
            scene_result: Results from descriptive scene analysis
            
        Returns:
            Formatted scene analysis context
        """
        context_parts = []
        
        # Add image caption if available
        if scene_result.get("image_caption"):
            context_parts.append(f"IMAGE CAPTION: {scene_result['image_caption']}")
            
        # Check if scene was found
        if not scene_result["scene_found"]:
            if scene_result.get("image_caption"):
                return f"IMAGE CAPTION: {scene_result['image_caption']}\n\nNo clear medical scene elements were identified in the image beyond what is described in the caption."
            else:
                return "No clear scene elements were identified in the image."
        
        # Add scene summary if available
        if scene_result["scene_summary"]:
            context_parts.append("SCENE OVERVIEW:")
            # Check for caption-based summary
            if "caption" in scene_result["scene_summary"]:
                context_parts.append(f"- {scene_result['scene_summary']['caption']}")
            else:
                for category, summary in scene_result["scene_summary"].items():
                    context_parts.append(f"- {summary}")
        
        # Determine which aspects to focus on based on query focus
        query_focus = scene_result.get("query_focus", "general")
        
        # Add detailed information based on query focus
        context_parts.append("\nDETAILED ANALYSIS:")
        
        # If scene overview is requested or query is general
        if query_focus in ["scene_overview", "general"]:
            # Include all key elements
            self._add_scene_elements(context_parts, scene_result)
        
        # If specific focus is requested, prioritize that information
        elif query_focus == "procedure":
            # Procedure details
            if scene_result["procedures"]:
                context_parts.append("Medical Procedures:")
                for proc in scene_result["procedures"]:
                    context_parts.append(f"- {proc['name']}")
                    for attr, value in proc["attributes"].items():
                        context_parts.append(f"  - {attr}: {value}")
                
                # Add procedure hierarchy if available
                if scene_result["scene_hierarchy"]:
                    context_parts.append("\nProcedure Details:")
                    for proc_name, details in scene_result["scene_hierarchy"].items():
                        context_parts.append(f"- {proc_name}:")
                        if details["performed_by"]:
                            context_parts.append(f"  - Performed by: {', '.join(details['performed_by'])}")
                        if details["performed_on"]:
                            context_parts.append(f"  - Performed on: {', '.join(details['performed_on'])}")
                        if details["using_equipment"]:
                            context_parts.append(f"  - Using equipment: {', '.join(details['using_equipment'])}")
                        if details["related_findings"]:
                            context_parts.append(f"  - Related findings: {', '.join(details['related_findings'])}")
            else:
                context_parts.append("No specific medical procedures were identified in the image.")
            
            # Include equipment as it's related to procedures
            if scene_result["equipment"]:
                context_parts.append("\nMedical Equipment Present:")
                for equip in scene_result["equipment"]:
                    context_parts.append(f"- {equip['name']}")
        
        elif query_focus == "equipment":
            # Equipment details
            if scene_result["equipment"]:
                context_parts.append("Medical Equipment:")
                for equip in scene_result["equipment"]:
                    context_parts.append(f"- {equip['name']}")
                    for attr, value in equip["attributes"].items():
                        context_parts.append(f"  - {attr}: {value}")
                
                # Add equipment relationships
                context_parts.append("\nEquipment Usage:")
                for rel in scene_result["relationships"]:
                    equip_ids = [e["id"] for e in scene_result["equipment"]]
                    if rel["source"] in equip_ids or rel["target"] in equip_ids:
                        context_parts.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
            else:
                context_parts.append("No specific medical equipment was identified in the image.")
        
        elif query_focus == "patient":
            # Patient details
            if scene_result["patients"] or scene_result["clinical_findings"]:
                if scene_result["patients"]:
                    context_parts.append("Patient Anatomy:")
                    for part in scene_result["patients"]:
                        context_parts.append(f"- {part['name']}")
                        for attr, value in part["attributes"].items():
                            context_parts.append(f"  - {attr}: {value}")
                
                if scene_result["clinical_findings"]:
                    context_parts.append("\nClinical Findings:")
                    for finding in scene_result["clinical_findings"]:
                        context_parts.append(f"- {finding['name']}")
                        for attr, value in finding["attributes"].items():
                            context_parts.append(f"  - {attr}: {value}")
            else:
                context_parts.append("No specific patient details or clinical findings were identified in the image.")
        
        elif query_focus == "personnel":
            # Personnel details
            if scene_result["personnel"]:
                context_parts.append("Healthcare Personnel:")
                for person in scene_result["personnel"]:
                    context_parts.append(f"- {person['name']}")
                    for attr, value in person["attributes"].items():
                        context_parts.append(f"  - {attr}: {value}")
                
                # Add personnel relationships
                context_parts.append("\nPersonnel Actions:")
                for rel in scene_result["relationships"]:
                    personnel_ids = [p["id"] for p in scene_result["personnel"]]
                    if rel["source"] in personnel_ids:
                        context_parts.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
            else:
                context_parts.append("No specific healthcare personnel were identified in the image.")
        
        # Add key relationships if not already covered
        if query_focus not in ["procedure", "equipment", "personnel"]:
            self._add_relationships(context_parts, scene_result)
        
        # Join all context parts with newlines
        return "\n".join(context_parts) 
    
    def _format_reasoning_traversal(self, result: Dict[str, Any]) -> str:
        """Format reasoning traversal results into a structured context for the LLM.
        
        Args:
            result: Results from reasoning traversal
            
        Returns:
            Formatted reasoning context
        """
        context_parts = []
        
        # Add image caption if available
        if result.get("image_caption"):
            context_parts.append(f"IMAGE CAPTION: {result['image_caption']}")
        
        # Add graph overview
        context_parts.append("KNOWLEDGE GRAPH OVERVIEW:")
        graph_structure = result["graph_structure"]
        context_parts.append(f"- Contains {graph_structure['nodes_count']} elements and {graph_structure['edges_count']} relationships")
        
        node_types = graph_structure.get("node_types", [])
        if node_types:
            context_parts.append(f"- Types of elements: {', '.join(node_types)}")
        
        relation_types = graph_structure.get("relation_types", [])
        if relation_types:
            context_parts.append(f"- Types of relationships: {', '.join(relation_types)}")
        
        # Add central elements section
        central_elements = result.get("central_elements", [])
        if central_elements:
            context_parts.append("\nCENTRAL ELEMENTS IN THE SCENE:")
            for element in central_elements:
                context_parts.append(f"- {element['name']} ({element['type']})")
                # Add key attributes
                for attr, value in element.get("attributes", {}).items():
                    context_parts.append(f"  - {attr}: {value}")
        
        # Format all elements by type
        scene_elements = result.get("scene_elements", {})
        if scene_elements:
            context_parts.append("\nSCENE ELEMENTS BY TYPE:")
            
            # Order types in a meaningful way - setting first, then personnel, etc.
            type_order = ["scene_setting", "personnel", "procedures", "equipment", "clinical_elements"]
            processed_types = set()
            
            # First add types in the preferred order
            for element_type in type_order:
                if element_type in scene_elements:
                    elements = scene_elements[element_type]
                    type_display = element_type.replace("_", " ").title()
                    context_parts.append(f"\n{type_display}:")
                    
                    for element in elements:
                        context_parts.append(f"- {element['name']}")
                        # Add significant attributes (limit to keep context focused)
                        significant_attrs = list(element.get("attributes", {}).items())[:3]
                        for attr, value in significant_attrs:
                            context_parts.append(f"  - {attr}: {value}")
                    
                    processed_types.add(element_type)
            
            # Add any remaining types
            for element_type, elements in scene_elements.items():
                if element_type not in processed_types:
                    type_display = element_type.replace("_", " ").title()
                    context_parts.append(f"\n{type_display}:")
                    
                    for element in elements:
                        context_parts.append(f"- {element['name']}")
                        # Add significant attributes (limit to keep context focused)
                        significant_attrs = list(element.get("attributes", {}).items())[:3]
                        for attr, value in significant_attrs:
                            context_parts.append(f"  - {attr}: {value}")
        
        # Add key relationships
        key_relationships = result.get("key_relationships", [])
        if key_relationships:
            context_parts.append("\nKEY RELATIONSHIPS:")
            # Limit to most significant relationships (if there are many)
            max_relationships = 15
            if len(key_relationships) > max_relationships:
                context_parts.append(f"(Showing {max_relationships} out of {len(key_relationships)} relationships)")
                key_relationships = key_relationships[:max_relationships]
            
            for rel in key_relationships:
                context_parts.append(f"- {rel['source_name']} {rel['relation']} {rel['target_name']}")
        
        # Add identified patterns
        graph_patterns = result.get("graph_patterns", [])
        if graph_patterns:
            context_parts.append("\nIDENTIFIED PATTERNS:")
            
            for pattern in graph_patterns:
                pattern_type = pattern["pattern_type"].replace("_", " ").title()
                context_parts.append(f"\n{pattern_type}:")
                
                instances = pattern["instances"]
                if pattern_type == "Action Chains" and isinstance(instances, list):
                    for instance in instances:
                        actor = instance.get("actor", "")
                        action = instance.get("action", "")
                        targets = instance.get("targets", [])
                        
                        if targets:
                            target_str = ", ".join(targets)
                            context_parts.append(f"- {actor} {action} {target_str}")
                        else:
                            context_parts.append(f"- {actor} {action}")
                            
                elif pattern_type == "Object Hierarchies" and isinstance(instances, list):
                    for instance in instances:
                        root = instance.get("root", "")
                        children = instance.get("children", [])
                        child_str = ", ".join(children)
                        
                        context_parts.append(f"- {root} has parts: {child_str}")
                
                elif pattern_type == "Attribute Clusters" and isinstance(instances, dict):
                    for attr, entities in instances.items():
                        context_parts.append(f"- Shared attribute '{attr}':")
                        for entity in entities:
                            name = entity.get("entity", "")
                            value = entity.get("value", "")
                            context_parts.append(f"  - {name}: {value}")
                            
                elif isinstance(instances, list):
                    # Generic handling for other pattern types
                    for i, instance in enumerate(instances):
                        context_parts.append(f"- Instance {i+1}: {instance}")
                elif isinstance(instances, dict):
                    # Generic handling for dictionary-based patterns
                    for key, value in instances.items():
                        context_parts.append(f"- {key}: {value}")
        
        # Join all context parts with newlines
        return "\n".join(context_parts) 