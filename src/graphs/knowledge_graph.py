"""Knowledge graph construction and querying for Medical VQA."""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import os

from src.config import MIN_GROUNDING_SCORE, NODE_SIMILARITY_THRESHOLD, VISUALIZATION_DIR

logger = logging.getLogger(__name__)


class MedicalKnowledgeGraph:
    """Knowledge graph for representing and querying medical concepts from images."""
    
    def __init__(self, similarity_model: str = "all-MiniLM-L6-v2"):
        """Initialize the knowledge graph.
        
        Args:
            similarity_model: Name of the sentence transformer model for node similarity
        """
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Load sentence transformer for semantic similarity
        try:
            self.similarity_model = SentenceTransformer(similarity_model)
        except Exception as e:
            logger.error(f"Failed to load similarity model: {e}")
            self.similarity_model = None
        
        # Minimum grounding score to include in graph
        self.min_grounding_score = MIN_GROUNDING_SCORE
        self.similarity_threshold = NODE_SIMILARITY_THRESHOLD
        
    def build_from_concepts(
        self, 
        concepts: Dict[str, Any], 
        image_id: str
    ) -> None:
        """Build knowledge graph from extracted concepts.
        
        Args:
            concepts: Dictionary of extracted concepts with grounding scores
            image_id: Identifier for the source image
        """
        # Clear any existing graph
        self.graph.clear()
        
        # Add global image node
        self.graph.add_node(
            f"image:{image_id}", 
            type="image",
            id=image_id
        )
        
        # Process anatomical structures
        if "anatomical_structures" in concepts:
            self._add_structures(concepts["anatomical_structures"], image_id)
        
        # Process findings/abnormalities
        if "findings" in concepts:
            self._add_findings(concepts["findings"], image_id)
        
        # Process relationships
        if "relationships" in concepts:
            self._add_relationships(concepts["relationships"])
        
        # Log graph statistics
        logger.info(f"Built knowledge graph with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} edges")
    
    def _add_structures(
        self, 
        structures: List[Dict[str, Any]], 
        image_id: str
    ) -> None:
        """Add anatomical structures to the graph.
        
        Args:
            structures: List of anatomical structures with attributes
            image_id: Identifier for the source image
        """
        for structure in structures:
            # Skip if grounding score is below threshold
            if "grounding_score" in structure and structure["grounding_score"] < self.min_grounding_score:
                logger.info(f"Skipping structure {structure['name']} due to low grounding score: "
                           f"{structure.get('grounding_score', 0)}")
                continue
            
            # Create node ID
            node_id = f"structure:{structure['name']}"
            
            # Check for potential duplicates via semantic similarity
            similar_node = self._find_similar_node(structure['name'], "structure")
            if similar_node:
                logger.info(f"Found similar structure node: {similar_node} for {structure['name']}")
                node_id = similar_node
            
            # Add node with all attributes
            self.graph.add_node(
                node_id,
                type="anatomical_structure",
                name=structure["name"],
                grounding_score=structure.get("grounding_score", 1.0),
                **structure.get("attributes", {})
            )
            
            # Connect to image
            self.graph.add_edge(
                f"image:{image_id}",
                node_id,
                relation="contains"
            )
    
    def _add_findings(
        self, 
        findings: List[Dict[str, Any]], 
        image_id: str
    ) -> None:
        """Add medical findings to the graph.
        
        Args:
            findings: List of medical findings with attributes
            image_id: Identifier for the source image
        """
        for finding in findings:
            # Skip if grounding score is below threshold
            if "grounding_score" in finding and finding["grounding_score"] < self.min_grounding_score:
                logger.info(f"Skipping finding {finding['name']} due to low grounding score: "
                           f"{finding.get('grounding_score', 0)}")
                continue
            
            # Create node ID
            node_id = f"finding:{finding['name']}"
            
            # Check for potential duplicates via semantic similarity
            similar_node = self._find_similar_node(finding['name'], "finding")
            if similar_node:
                logger.info(f"Found similar finding node: {similar_node} for {finding['name']}")
                node_id = similar_node
            
            # Add node with all attributes
            self.graph.add_node(
                node_id,
                type="finding",
                name=finding["name"],
                grounding_score=finding.get("grounding_score", 1.0),
                **finding.get("attributes", {})
            )
            
            # Connect to image
            self.graph.add_edge(
                f"image:{image_id}",
                node_id,
                relation="shows"
            )
    
    def _add_relationships(
        self, 
        relationships: List[Dict[str, Any]]
    ) -> None:
        """Add relationships between entities in the graph.
        
        Args:
            relationships: List of relationships between entities
        """
        for rel in relationships:
            # Skip if grounding score is below threshold
            if "grounding_score" in rel and rel["grounding_score"] < self.min_grounding_score:
                logger.info(f"Skipping relationship {rel['subject']} -> {rel['object']} "
                           f"due to low grounding score: {rel.get('grounding_score', 0)}")
                continue
            
            # Find subject and object nodes (including similar nodes)
            subject_prefix = self._determine_node_type_prefix(rel['subject'])
            object_prefix = self._determine_node_type_prefix(rel['object'])
            
            subject_node = f"{subject_prefix}:{rel['subject']}"
            object_node = f"{object_prefix}:{rel['object']}"
            
            # Find similar nodes if exact ones don't exist
            if not self.graph.has_node(subject_node):
                similar_subject = self._find_similar_node(rel['subject'])
                if similar_subject:
                    subject_node = similar_subject
                else:
                    logger.warning(f"Subject node {subject_node} not found, skipping relationship")
                    continue
            
            if not self.graph.has_node(object_node):
                similar_object = self._find_similar_node(rel['object'])
                if similar_object:
                    object_node = similar_object
                else:
                    logger.warning(f"Object node {object_node} not found, skipping relationship")
                    continue
            
            # Add the relationship edge
            self.graph.add_edge(
                subject_node,
                object_node,
                relation=rel["relation"],
                grounding_score=rel.get("grounding_score", 1.0)
            )
    
    def _determine_node_type_prefix(self, entity_name: str) -> str:
        """Determine the likely type prefix for an entity name.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Prefix string ('structure', 'finding', etc.)
        """
        # Check if any existing nodes contain this name
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', '')
            if node_type in ['anatomical_structure', 'finding'] and entity_name.lower() in node.lower():
                return 'structure' if node_type == 'anatomical_structure' else 'finding'
        
        # Default to structure if we can't determine
        return 'structure'
    
    def _find_similar_node(
        self, 
        name: str, 
        node_type: Optional[str] = None
    ) -> Optional[str]:
        """Find semantically similar node in the graph.
        
        Args:
            name: Name to find similar matches for
            node_type: Optional filter by node type
            
        Returns:
            ID of the most similar node, or None if no match
        """
        if not self.similarity_model or not self.graph.nodes:
            return None
        
        # Get all candidate nodes of the requested type
        candidates = []
        for node in self.graph.nodes:
            if node_type and node_type not in node:
                continue
            candidates.append(node)
        
        if not candidates:
            return None
        
        try:
            # Get node names for comparison
            node_names = [self.graph.nodes[node].get('name', '') for node in candidates]
            
            # Calculate embeddings
            query_embedding = self.similarity_model.encode([name], convert_to_numpy=True)
            node_embeddings = self.similarity_model.encode(node_names, convert_to_numpy=True)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, node_embeddings.T)[0]
            
            # Find best match above threshold
            best_idx = np.argmax(similarities)
            if similarities[best_idx] >= self.similarity_threshold:
                return candidates[best_idx]
        
        except Exception as e:
            logger.error(f"Error calculating node similarity: {e}")
        
        return None
    
    def traverse(
        self, 
        query: str, 
        strategy: str = "subgraph_extraction",
        image_caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Traverse the graph according to the query and strategy.
        
        Args:
            query: The query or concept to look for
            strategy: Name of the traversal strategy to use
            image_caption: Optional caption of the image for better context
            
        Returns:
            Results of the traversal based on strategy
        """
        # For descriptive queries, prioritize descriptive_scene_analysis
        if strategy == "descriptive_scene_analysis":
            result = self._analyze_scene(query, image_caption)
            
            # If scene analysis finds no meaningful elements and we have an image caption,
            # create a minimal graph with the caption as context
            if not result["scene_found"] and image_caption:
                result["scene_found"] = True
                result["scene_summary"] = {"caption": f"Image shows: {image_caption}"}
                result["image_caption"] = image_caption
                
            return result
            
        elif strategy == "node_existence":
            return self._check_node_existence(query)
        elif strategy == "attribute_retrieval":
            return self._retrieve_attributes(query)
        elif strategy == "path_finding":
            return self._find_paths(query)
        elif strategy == "comparative_traversal":
            return self._compare_nodes(query)
        elif strategy == "procedure_evaluation":
            return self._evaluate_procedure(query)
        elif strategy == "reasoning_traversal":
            # New strategy that focuses on relationship-based reasoning
            return self._reason_with_graph(query, image_caption)
        else:  # Default to subgraph extraction
            result = self._extract_subgraph(query)
            
            # For empty subgraphs with an image caption, add caption information
            if (not result["found"] or not result.get("nodes")) and image_caption:
                result["image_caption"] = image_caption
                result["caption_based"] = True
                
            return result
    
    def _check_node_existence(self, query: str) -> Dict[str, Any]:
        """Check if a concept exists in the graph.
        
        Args:
            query: Concept name to check for
            
        Returns:
            Dictionary with existence information
        """
        # Try to find a node with the query term
        matching_nodes = []
        for node in self.graph.nodes:
            node_name = self.graph.nodes[node].get('name', '')
            if query.lower() in node_name.lower():
                matching_nodes.append(node)
        
        # If no exact match, try semantic similarity
        if not matching_nodes and self.similarity_model:
            similar_node = self._find_similar_node(query)
            if similar_node:
                matching_nodes.append(similar_node)
        
        # Prepare result
        result = {
            "exists": len(matching_nodes) > 0,
            "matching_nodes": [],
            "grounding_scores": []
        }
        
        # Add details for all matches
        for node in matching_nodes:
            node_data = self.graph.nodes[node]
            result["matching_nodes"].append({
                "id": node,
                "name": node_data.get('name', ''),
                "type": node_data.get('type', ''),
                "attributes": {k: v for k, v in node_data.items() 
                              if k not in ['name', 'type', 'grounding_score']}
            })
            result["grounding_scores"].append(node_data.get('grounding_score', 0.0))
        
        return result
    
    def _retrieve_attributes(self, query: str) -> Dict[str, Any]:
        """Retrieve attributes of nodes matching the query.
        
        Args:
            query: Concept to retrieve attributes for
            
        Returns:
            Dictionary with node attributes
        """
        # First find matching nodes
        existence_result = self._check_node_existence(query)
        
        result = {
            "found": existence_result["exists"],
            "attributes": [],
            "relationships": []
        }
        
        # For each matching node, collect attributes and relationships
        for node_info in existence_result["matching_nodes"]:
            node_id = node_info["id"]
            
            # Add node attributes
            result["attributes"].append({
                "node": node_info["name"],
                "type": node_info["type"],
                "properties": node_info["attributes"]
            })
            
            # Add outgoing relationships
            for _, target, data in self.graph.out_edges(node_id, data=True):
                target_name = self.graph.nodes[target].get('name', target)
                result["relationships"].append({
                    "from": node_info["name"],
                    "relation": data.get('relation', 'connected_to'),
                    "to": target_name,
                    "grounding_score": data.get('grounding_score', 1.0)
                })
            
            # Add incoming relationships
            for source, _, data in self.graph.in_edges(node_id, data=True):
                source_name = self.graph.nodes[source].get('name', source)
                if "image:" not in source:  # Skip image container relationships
                    result["relationships"].append({
                        "from": source_name,
                        "relation": data.get('relation', 'connected_to'),
                        "to": node_info["name"],
                        "grounding_score": data.get('grounding_score', 1.0)
                    })
        
        return result
    
    def _find_paths(self, query: str) -> Dict[str, Any]:
        """Find paths between entities mentioned in the query.
        
        Args:
            query: Query containing entities to find paths between
            
        Returns:
            Dictionary with path information
        """
        # This is a simplified implementation
        # A more sophisticated version would parse the query to extract source and target
        
        # For demo, we'll split the query on "and", "to", or "between"
        import re
        entities = re.split(r'and|to|between', query)
        entities = [e.strip() for e in entities if e.strip()]
        
        result = {
            "paths_found": False,
            "paths": []
        }
        
        # Need at least two entities for a path
        if len(entities) < 2:
            return result
        
        # Find nodes for the first and second entity
        source_node = self._find_similar_node(entities[0])
        target_node = self._find_similar_node(entities[1])
        
        if not source_node or not target_node:
            return result
        
        # Find all simple paths between the nodes
        try:
            paths = list(nx.all_simple_paths(self.graph, source_node, target_node, cutoff=3))
            # Add paths in reverse direction if none found
            if not paths:
                paths = list(nx.all_simple_paths(self.graph, target_node, source_node, cutoff=3))
            
            if paths:
                result["paths_found"] = True
                
                # Format each path
                for path in paths:
                    path_info = {
                        "nodes": [],
                        "relations": []
                    }
                    
                    # Add node details
                    for node in path:
                        node_data = self.graph.nodes[node]
                        path_info["nodes"].append({
                            "id": node,
                            "name": node_data.get('name', node),
                            "type": node_data.get('type', '')
                        })
                    
                    # Add relationship details
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        path_info["relations"].append({
                            "from": self.graph.nodes[path[i]].get('name', path[i]),
                            "relation": edge_data.get('relation', 'connected_to'),
                            "to": self.graph.nodes[path[i+1]].get('name', path[i+1])
                        })
                    
                    result["paths"].append(path_info)
        
        except Exception as e:
            logger.error(f"Error finding paths: {e}")
        
        return result
    
    def _compare_nodes(self, query: str) -> Dict[str, Any]:
        """Compare attributes between nodes mentioned in the query.
        
        Args:
            query: Query containing entities to compare
            
        Returns:
            Dictionary with comparison information
        """
        # Similar to path finding, extract entities to compare
        import re
        entities = re.split(r'compare|versus|vs|and|between', query)
        entities = [e.strip() for e in entities if e.strip()]
        
        result = {
            "comparison_possible": False,
            "entities": [],
            "common_attributes": [],
            "differences": []
        }
        
        # Need at least two entities for comparison
        if len(entities) < 2:
            return result
        
        # Find nodes for all entities
        entity_nodes = []
        for entity in entities:
            node = self._find_similar_node(entity)
            if node:
                entity_nodes.append(node)
        
        if len(entity_nodes) < 2:
            return result
        
        # Get node data for comparison
        node_data = []
        for node in entity_nodes:
            data = self.graph.nodes[node]
            node_data.append({
                "id": node,
                "name": data.get('name', node),
                "type": data.get('type', ''),
                "attributes": {k: v for k, v in data.items() 
                              if k not in ['name', 'type', 'grounding_score']}
            })
        
        result["comparison_possible"] = True
        result["entities"] = node_data
        
        # Find common attributes
        common_keys = set.intersection(
            *[set(node["attributes"].keys()) for node in node_data]
        )
        
        for key in common_keys:
            values = [node["attributes"][key] for node in node_data]
            result["common_attributes"].append({
                "attribute": key,
                "values": values,
                "is_same": len(set(values)) == 1
            })
        
        # Find differences
        all_keys = set.union(
            *[set(node["attributes"].keys()) for node in node_data]
        )
        different_keys = all_keys - common_keys
        
        for key in different_keys:
            diff = {
                "attribute": key,
                "values": []
            }
            for node in node_data:
                diff["values"].append(node["attributes"].get(key, "N/A"))
            result["differences"].append(diff)
        
        return result
    
    def _extract_subgraph(self, query: str) -> Dict[str, Any]:
        """Extract a relevant subgraph around concepts in the query.
        
        Args:
            query: Query containing concepts of interest
            
        Returns:
            Dictionary with subgraph information
        """
        # Find matching nodes for query terms
        existence_result = self._check_node_existence(query)
        
        result = {
            "found": existence_result["exists"],
            "nodes": [],
            "edges": []
        }
        
        if not existence_result["exists"]:
            return result
        
        # Get seed nodes from matches
        seed_nodes = [node_info["id"] for node_info in existence_result["matching_nodes"]]
        
        # Build neighborhood subgraph (1-hop from seed nodes)
        neighbors = set(seed_nodes)
        for node in seed_nodes:
            neighbors.update(self.graph.successors(node))
            neighbors.update(self.graph.predecessors(node))
        
        # Create subgraph
        subgraph = self.graph.subgraph(neighbors)
        
        # Extract node information
        for node in subgraph.nodes:
            node_data = subgraph.nodes[node]
            # Skip image container nodes
            if "image:" in node:
                continue
                
            result["nodes"].append({
                "id": node,
                "name": node_data.get('name', node.split(':')[-1]),
                "type": node_data.get('type', ''),
                "grounding_score": node_data.get('grounding_score', 1.0),
                "attributes": {k: v for k, v in node_data.items() 
                              if k not in ['name', 'type', 'grounding_score']}
            })
        
        # Extract edge information
        for source, target, data in subgraph.edges(data=True):
            # Skip edges to/from image container
            if "image:" in source or "image:" in target:
                continue
                
            result["edges"].append({
                "source": source,
                "target": target,
                "relation": data.get('relation', 'connected_to'),
                "grounding_score": data.get('grounding_score', 1.0)
            })
        
        return result
    
    def to_json(self) -> str:
        """Convert the knowledge graph to a JSON string.
        
        Returns:
            JSON string representation of the graph
        """
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                **attrs
            }
            data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                **attrs
            }
            data["edges"].append(edge_data)
        
        return json.dumps(data, indent=2)
    
    def visualize(self, output_path: Optional[str] = None) -> str:
        """Visualize the knowledge graph.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        if not output_path:
            output_path = os.path.join(VISUALIZATION_DIR, "knowledge_graph.png")
        
        # Create a spring layout
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Node colors by type
        colors = []
        for node in self.graph.nodes():
            if "image:" in node:
                colors.append("lightblue")
            elif "structure:" in node:
                colors.append("lightgreen")
            elif "finding:" in node:
                colors.append("salmon")
            else:
                colors.append("gray")
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, alpha=0.8, node_size=500)
        
        # Edge colors by relation type
        edge_colors = []
        for _, _, attrs in self.graph.edges(data=True):
            relation = attrs.get('relation', '')
            if relation == 'contains':
                edge_colors.append('lightblue')
            elif relation == 'shows':
                edge_colors.append('salmon')
            elif relation == 'located_in':
                edge_colors.append('green')
            elif relation == 'adjacent_to':
                edge_colors.append('purple')
            else:
                edge_colors.append('gray')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=1.5, alpha=0.7)
        
        # Draw labels
        labels = {}
        for node in self.graph.nodes():
            if "image:" in node:
                labels[node] = "Image"
            else:
                labels[node] = self.graph.nodes[node].get('name', node.split(':')[-1])
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_family='sans-serif')
        
        # Save figure
        plt.title("Medical Knowledge Graph", fontsize=15)
        plt.axis('off')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _analyze_scene(self, query: str, image_caption: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the scene for descriptive queries about what's happening or depicted.
        
        This method provides a comprehensive analysis of the scene by organizing graph elements
        into meaningful categories aligned with emergency room VQA domain needs:
        - Scene setting and environment
        - People present and their roles
        - Medical procedures being performed
        - Equipment and tools in use
        - Patient condition and visible clinical findings
        - Spatial and temporal relationships within the scene
        
        Args:
            query: The descriptive query about the scene
            image_caption: Optional caption of the image for better context
            
        Returns:
            Dictionary with structured scene analysis
        """
        # First, extract entire graph for comprehensive analysis
        result = {
            "scene_found": False,
            "scene_summary": {},
            "setting": [],
            "personnel": [],
            "patients": [],
            "procedures": [],
            "equipment": [],
            "clinical_findings": [],
            "relationships": [],
            "scene_hierarchy": {},
            "query_focus": self._determine_query_focus(query),
            "image_caption": image_caption
        }
        
        # Collect all nodes and categorize them based on type
        all_nodes = list(self.graph.nodes(data=True))
        
        # If graph is empty but we have an image caption, still provide basic scene info
        if not all_nodes and image_caption:
            result["scene_found"] = True
            result["scene_summary"] = {"caption": f"The image shows: {image_caption}"}
            return result
            
        # Skip analysis if graph is empty and no caption
        if not all_nodes:
            return result
        
        # Mark that we found scene elements
        result["scene_found"] = True
        
        # Categorize nodes by their types
        for node_id, data in all_nodes:
            # Skip image container node
            if "image:" in node_id:
                continue
                
            node_type = data.get('type', '')
            node_name = data.get('name', node_id.split(':')[-1])
            node_info = {
                "id": node_id,
                "name": node_name,
                "grounding_score": data.get('grounding_score', 1.0),
                "attributes": {k: v for k, v in data.items() 
                             if k not in ['name', 'type', 'grounding_score']}
            }
            
            # Categorize by type with ER-specific categories
            if "anatomical_structure" in node_type:
                # Add to patients section if it's a body part
                result["patients"].append(node_info)
            elif "finding" in node_type:
                # Clinical findings are symptoms, conditions, injuries
                result["clinical_findings"].append(node_info)
            elif any(role in node_name.lower() for role in 
                    ["doctor", "nurse", "physician", "staff", "worker", "paramedic", "emt"]):
                # Healthcare personnel
                result["personnel"].append(node_info)
            elif any(eq in node_name.lower() for eq in 
                    ["monitor", "ventilator", "mask", "tube", "iv", "syringe", "equipment",
                     "device", "needle", "bandage", "dressing", "bed", "stretcher"]):
                # Medical equipment
                result["equipment"].append(node_info)
            elif any(proc in node_name.lower() for proc in 
                    ["procedure", "treatment", "injection", "intubation", "suturing", 
                     "resuscitation", "monitoring", "vaccination", "examination"]):
                # Medical procedures
                result["procedures"].append(node_info)
            elif any(setting in node_name.lower() for setting in 
                    ["room", "hospital", "ward", "icu", "er", "facility", "clinic"]):
                # Setting/environment
                result["setting"].append(node_info)
            else:
                # Determine category from attributes
                attrs = node_info["attributes"]
                if "location" in attrs or "position" in attrs:
                    result["setting"].append(node_info)
                elif "action" in attrs or "performing" in attrs:
                    result["procedures"].append(node_info)
                elif "status" in attrs or "condition" in attrs:
                    result["clinical_findings"].append(node_info)
        
        # Extract relationships between entities
        for source, target, data in self.graph.edges(data=True):
            # Skip edges from image container
            if "image:" in source:
                continue
                
            # Get source and target names
            source_name = self.graph.nodes[source].get('name', source.split(':')[-1])
            target_name = self.graph.nodes[target].get('name', target.split(':')[-1])
            
            # Add relationship
            result["relationships"].append({
                "source": source,
                "source_name": source_name,
                "target": target,
                "target_name": target_name,
                "relation": data.get('relation', 'connected_to'),
                "grounding_score": data.get('grounding_score', 1.0)
            })
        
        # Build scene hierarchy - organize relationships into a structured format
        hierarchy = {}
        
        # Add procedures with their related equipment and findings
        for proc_info in result["procedures"]:
            proc_id = proc_info["id"]
            proc_hierarchy = {
                "name": proc_info["name"],
                "performed_by": [],
                "performed_on": [],
                "using_equipment": [],
                "related_findings": []
            }
            
            # Find relationships involving this procedure
            for rel in result["relationships"]:
                # Procedure is performed by someone
                if rel["target"] == proc_id and rel["relation"] in ["performed_by", "conducted_by", "done_by"]:
                    proc_hierarchy["performed_by"].append(rel["source_name"])
                
                # Procedure is performed on someone/something
                if rel["source"] == proc_id and rel["relation"] in ["performed_on", "applied_to", "done_on"]:
                    proc_hierarchy["performed_on"].append(rel["target_name"])
                
                # Procedure uses equipment
                if rel["source"] == proc_id and rel["relation"] in ["uses", "requires", "utilizes"]:
                    proc_hierarchy["using_equipment"].append(rel["target_name"])
                
                # Procedure relates to findings
                if (rel["source"] == proc_id and rel["target"] in [f["id"] for f in result["clinical_findings"]]) or \
                   (rel["target"] == proc_id and rel["source"] in [f["id"] for f in result["clinical_findings"]]):
                    finding_name = rel["target_name"] if rel["source"] == proc_id else rel["source_name"]
                    proc_hierarchy["related_findings"].append(finding_name)
            
            # Add to hierarchy
            hierarchy[proc_info["name"]] = proc_hierarchy
        
        # Add the hierarchy to the result
        result["scene_hierarchy"] = hierarchy
        
        # Create a concise scene summary
        summary = {}
        
        # Setting summary
        if result["setting"]:
            setting_names = [s["name"] for s in result["setting"]]
            summary["setting"] = f"Scene takes place in/at {', '.join(setting_names)}"
        
        # Personnel summary
        if result["personnel"]:
            personnel_names = [p["name"] for p in result["personnel"]]
            summary["personnel"] = f"Healthcare personnel present: {', '.join(personnel_names)}"
        
        # Procedure summary
        if result["procedures"]:
            proc_names = [p["name"] for p in result["procedures"]]
            summary["procedures"] = f"Medical procedures being performed: {', '.join(proc_names)}"
        
        # Equipment summary
        if result["equipment"]:
            equip_names = [e["name"] for e in result["equipment"]]
            summary["equipment"] = f"Medical equipment visible: {', '.join(equip_names)}"
        
        # Clinical findings summary
        if result["clinical_findings"]:
            finding_names = [f["name"] for f in result["clinical_findings"]]
            summary["findings"] = f"Clinical findings: {', '.join(finding_names)}"
        
        # Add the summary to the result
        result["scene_summary"] = summary
        
        return result
    
    def _determine_query_focus(self, query: str) -> str:
        """Determine the focus of a descriptive query to prioritize relevant information.
        
        Args:
            query: The descriptive query
            
        Returns:
            Focus category string
        """
        query_lower = query.lower()
        
        # Scene/setting focus
        if any(term in query_lower for term in ["what is", "what does", "scene", "depict", 
                                               "show", "setting", "where"]):
            return "scene_overview"
        
        # Procedure focus
        elif any(term in query_lower for term in ["procedure", "treatment", "doing", "performed", 
                                               "technique", "how", "carried out"]):
            return "procedure"
            
        # Equipment focus
        elif any(term in query_lower for term in ["equipment", "tools", "using", "device", 
                                               "instruments", "machine"]):
            return "equipment"
            
        # Patient focus
        elif any(term in query_lower for term in ["patient", "person", "condition", "status", 
                                               "injured", "sick"]):
            return "patient"
            
        # Personnel focus
        elif any(term in query_lower for term in ["doctor", "nurse", "staff", "worker", 
                                               "personnel", "who"]):
            return "personnel"
            
        # Default to general overview
        return "general"
    
    def _evaluate_procedure(self, query: str) -> Dict[str, Any]:
        """Evaluate medical procedures for correctness or appropriateness.
        
        This method specializes in procedural queries common in emergency room contexts,
        such as "Is the technique proper?", "Is the PPE adequate?", "Is the procedure correct?".
        
        Args:
            query: Query about procedure correctness or appropriateness
            
        Returns:
            Dictionary with procedure evaluation data
        """
        # First, perform a comprehensive scene analysis
        scene_analysis = self._analyze_scene(query)
        
        result = {
            "evaluation_possible": scene_analysis["scene_found"],
            "procedure_identified": len(scene_analysis["procedures"]) > 0,
            "procedures": scene_analysis["procedures"],
            "related_equipment": scene_analysis["equipment"],
            "personnel_involved": scene_analysis["personnel"],
            "evaluation_focus": self._determine_evaluation_focus(query),
            "key_relationships": []
        }
        
        # Extract specific relationships for procedure evaluation
        if result["procedure_identified"]:
            # Find relationships relevant to procedures
            for rel in scene_analysis["relationships"]:
                # Check if relationship involves procedure
                proc_ids = [p["id"] for p in scene_analysis["procedures"]]
                if rel["source"] in proc_ids or rel["target"] in proc_ids:
                    result["key_relationships"].append(rel)
        
        return result
    
    def _determine_evaluation_focus(self, query: str) -> str:
        """Determine the focus of a procedural evaluation query.
        
        Args:
            query: The procedure evaluation query
            
        Returns:
            Focus category string
        """
        query_lower = query.lower()
        
        # Technique correctness
        if any(term in query_lower for term in ["technique", "correct", "proper", "right way", 
                                               "appropriate", "properly"]):
            return "technique"
        
        # Safety protocol
        elif any(term in query_lower for term in ["safety", "safe", "protocol", "guidelines", 
                                               "standards", "regulations"]):
            return "safety"
        
        # Equipment usage
        elif any(term in query_lower for term in ["equipment", "using", "usage", "utilized", 
                                               "tools", "devices"]):
            return "equipment_usage"
        
        # PPE and protection
        elif any(term in query_lower for term in ["ppe", "protection", "mask", "gloves", 
                                               "gown", "shield", "protective"]):
            return "ppe"
        
        # Procedure steps
        elif any(term in query_lower for term in ["steps", "process", "sequence", "order", 
                                               "following"]):
            return "procedure_steps"
        
        # Default to general evaluation
        return "general_correctness"
    
    def _reason_with_graph(
        self, 
        query: str,
        image_caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reason with the graph structure to answer the query.
        
        This strategy analyzes the entire graph structure and relationships
        to provide a comprehensive answer beyond simple traversal. It's
        particularly useful for complex or descriptive queries.
        
        Args:
            query: The query to reason about
            image_caption: Optional caption for additional context
            
        Returns:
            Dictionary with reasoning results
        """
        result = {
            "reasoning_applied": True,
            "graph_structure": {
                "nodes_count": self.graph.number_of_nodes(),
                "edges_count": self.graph.number_of_edges(),
                "node_types": list(set([data.get('type', 'unknown') 
                                      for _, data in self.graph.nodes(data=True)])),
                "relation_types": list(set([data.get('relation', 'unknown') 
                                          for _, _, data in self.graph.edges(data=True)]))
            },
            "image_caption": image_caption,
            "scene_elements": {},
            "key_relationships": [],
            "central_elements": [],
            "query_focus": self._determine_query_focus(query)
        }
        
        # Extract all nodes by type
        node_by_type = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type not in node_by_type:
                node_by_type[node_type] = []
            
            node_by_type[node_type].append({
                "id": node,
                "name": data.get('name', node.split(':')[-1]),
                "attributes": {k: v for k, v in data.items() 
                              if k not in ['name', 'type', 'grounding_score']}
            })
        
        # Add all node types to result
        result["scene_elements"] = node_by_type
        
        # Extract key relationships (all edges)
        for source, target, data in self.graph.edges(data=True):
            source_name = self.graph.nodes[source].get('name', source.split(':')[-1])
            target_name = self.graph.nodes[target].get('name', target.split(':')[-1])
            relation = data.get('relation', 'related_to')
            
            result["key_relationships"].append({
                "source": source,
                "source_name": source_name,
                "target": target,
                "target_name": target_name,
                "relation": relation,
                "grounding_score": data.get('grounding_score', 1.0)
            })
        
        # Find central elements using node centrality
        try:
            # Calculate betweenness centrality to identify key nodes
            centrality = nx.betweenness_centrality(self.graph)
            central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Include top 5 central nodes (or fewer if graph is smaller)
            for node, score in central_nodes[:min(5, len(central_nodes))]:
                if score > 0:  # Only include nodes with some centrality
                    node_data = self.graph.nodes[node]
                    result["central_elements"].append({
                        "id": node,
                        "name": node_data.get('name', node.split(':')[-1]),
                        "type": node_data.get('type', 'unknown'),
                        "centrality_score": score,
                        "attributes": {k: v for k, v in node_data.items() 
                                      if k not in ['name', 'type', 'grounding_score']}
                    })
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
        
        # Identify graph patterns relevant to the query
        result["graph_patterns"] = self._identify_graph_patterns(query)
        
        return result
    
    def _identify_graph_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Identify relevant patterns in the graph structure based on the query.
        
        Args:
            query: The query to identify patterns for
            
        Returns:
            List of identified patterns with their elements
        """
        patterns = []
        
        # Try to identify common patterns in the graph
        try:
            # Pattern 1: Action chains (procedures performed by someone on something)
            action_chains = []
            
            for source, target, data in self.graph.edges(data=True):
                relation = data.get('relation', '')
                if relation in ['performs', 'conducts', 'executes', 'does']:
                    # This is an actor -> action relationship
                    actor = self.graph.nodes[source].get('name', source.split(':')[-1])
                    action = self.graph.nodes[target].get('name', target.split(':')[-1])
                    
                    # Check if this action is performed on something
                    action_targets = []
                    for s, t, d in self.graph.edges(data=True):
                        if s == target and d.get('relation', '') in ['performed_on', 'applied_to', 'targets']:
                            target_obj = self.graph.nodes[t].get('name', t.split(':')[-1])
                            action_targets.append(target_obj)
                    
                    action_chains.append({
                        "actor": actor,
                        "action": action,
                        "targets": action_targets
                    })
            
            if action_chains:
                patterns.append({
                    "pattern_type": "action_chains",
                    "instances": action_chains
                })
            
            # Pattern 2: Object hierarchies (part-of relationships)
            hierarchies = []
            part_of_edges = [(s, t) for s, t, d in self.graph.edges(data=True) 
                             if d.get('relation', '') in ['part_of', 'component_of', 'belongs_to']]
            
            if part_of_edges:
                # Build trees from part-of relationships
                parents = {}
                for child, parent in part_of_edges:
                    parents[child] = parent
                
                # Find roots (nodes that are not children)
                roots = set(parents.values()) - set(parents.keys())
                
                # Build hierarchies from roots
                for root in roots:
                    hierarchy = {
                        "root": self.graph.nodes[root].get('name', root.split(':')[-1]),
                        "children": []
                    }
                    
                    # Find direct children
                    for child, parent in parents.items():
                        if parent == root:
                            hierarchy["children"].append(
                                self.graph.nodes[child].get('name', child.split(':')[-1])
                            )
                    
                    hierarchies.append(hierarchy)
                
                patterns.append({
                    "pattern_type": "object_hierarchies",
                    "instances": hierarchies
                })
            
            # Pattern 3: Attribute clusters (entities with similar attributes)
            if len(self.graph.nodes) > 1:
                attribute_clusters = {}
                
                for node, data in self.graph.nodes(data=True):
                    for attr, value in data.items():
                        if attr not in ['name', 'type', 'id', 'grounding_score']:
                            if attr not in attribute_clusters:
                                attribute_clusters[attr] = []
                            
                            attribute_clusters[attr].append({
                                "entity": data.get('name', node.split(':')[-1]),
                                "value": value
                            })
                
                # Keep only attributes shared by multiple entities
                shared_attrs = {k: v for k, v in attribute_clusters.items() if len(v) > 1}
                
                if shared_attrs:
                    patterns.append({
                        "pattern_type": "attribute_clusters",
                        "instances": shared_attrs
                    })
            
        except Exception as e:
            logger.error(f"Error identifying graph patterns: {e}")
        
        return patterns 