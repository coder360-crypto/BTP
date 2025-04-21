"""NLP utility functions for emergency room medical question analysis."""

import re
import spacy
import nltk
import numpy as np
from typing import List, Dict, Set, Any, Tuple, Optional, Union
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Explicitly download punkt_tab if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # First make sure punkt is available
    nltk.download('punkt', quiet=True)
    # Then try to access punkt_tab again - it should be included in punkt package
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # If still missing, try a more explicit download
        nltk.download('punkt', download_dir=nltk.data.path[0], quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, use CLI command to install
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Emergency room medical terms for enhanced detection - tailored for visual QA
MEDICAL_TERMS = {
    # Body parts and anatomical structures commonly seen in ER images
    'anatomical': [
        'head', 'face', 'eye', 'eyes', 'nose', 'mouth', 'ear', 'ears', 'throat', 'neck',
        'chest', 'abdomen', 'arm', 'arms', 'hand', 'hands', 'finger', 'fingers', 'thumb',
        'leg', 'legs', 'foot', 'feet', 'toe', 'toes', 'ankle', 'knee', 'hip', 'shoulder',
        'elbow', 'wrist', 'back', 'spine', 'skull', 'rib', 'ribs', 'pelvis', 'joint',
        'forehead', 'temple', 'jaw', 'tongue', 'nostril', 'scalp', 'limb', 'limbs'
    ],
    
    # Injuries, conditions, and clinical findings visible in ER images
    'clinical_findings': [
        'laceration', 'cut', 'wound', 'injury', 'trauma', 'fracture', 'break', 'broken',
        'bruise', 'contusion', 'swelling', 'inflammation', 'bleeding', 'blood', 'rash',
        'burn', 'abrasion', 'scratch', 'puncture', 'deformity', 'dislocation', 'sprain',
        'strain', 'concussion', 'head injury', 'trauma', 'unconscious', 'conscious',
        'alert', 'pain', 'discomfort', 'discoloration', 'infection', 'fluid', 'discharge',
        'shock', 'paralysis', 'immobile', 'mobility', 'unstable', 'stable', 'critical'
    ],
    
    # Medical equipment and devices visible in ER photos
    'equipment': [
        'mask', 'oxygen', 'tube', 'tubing', 'iv', 'drip', 'infusion', 'catheter', 'monitor',
        'ventilator', 'intubation', 'cannula', 'nasal cannula', 'bed', 'stretcher', 'gurney',
        'wheelchair', 'crutch', 'crutches', 'bandage', 'gauze', 'dressing', 'cast', 'splint',
        'brace', 'collar', 'sling', 'ecg', 'ekg', 'defibrillator', 'bp cuff', 'blood pressure',
        'pulse oximeter', 'stethoscope', 'syringe', 'needle', 'suction', 'suture', 'gloves',
        'ppe', 'gown', 'face shield', 'goggles', 'saline', 'bag', 'pump', 'instrument', 'tray',
        'machine', 'device', 'cart', 'scan', 'imaging', 'x-ray', 'ultrasound', 'ct'
    ],
    
    # Procedural and safety terms related to ER protocols visible in images
    'procedure': [
        'intubation', 'cpr', 'resuscitation', 'airway', 'breathing', 'circulation', 'vitals',
        'examination', 'assessment', 'triage', 'treatment', 'injection', 'medication', 'admin',
        'administration', 'iv access', 'suturing', 'sterilization', 'disinfection', 'cleaning',
        'protocol', 'procedure', 'guideline', 'safety', 'precaution', 'isolation', 'sample',
        'collection', 'test', 'screening', 'monitoring', 'observation', 'imaging', 'scan',
        'x-ray', 'ultrasound', 'ct scan', 'mri', 'surgery', 'preparation', 'emergency', 'urgent',
        'standard', 'technique', 'proper', 'correct', 'appropriate', 'vaccination', 'injection'
    ],
    
    # Personnel and setting terms relevant to emergency room images
    'setting': [
        'doctor', 'physician', 'nurse', 'paramedic', 'emt', 'staff', 'healthcare worker',
        'patient', 'attendant', 'specialist', 'technician', 'caregiver', 'emergency room',
        'er', 'hospital', 'clinic', 'ward', 'icu', 'facility', 'isolation', 'ambulance',
        'hallway', 'room', 'bed', 'area', 'zone', 'station', 'department', 'center', 'unit'
    ]
}

# Lazy loading for evaluation models
_sentence_transformer = None
_nli_model = None
_nli_tokenizer = None
_rouge_scorer = None
_clip_model = None
_clip_processor = None

def _get_sentence_transformer():
    """Lazy loading for SentenceBERT model."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceBERT model: {e}")
            raise
    return _sentence_transformer

def _get_nli_model_and_tokenizer():
    """Lazy loading for NLI model and tokenizer."""
    global _nli_model, _nli_tokenizer
    if _nli_model is None or _nli_tokenizer is None:
        try:
            _nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
            _nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-roberta-base")
        except Exception as e:
            print(f"Error loading NLI model: {e}")
            raise
    return _nli_model, _nli_tokenizer

def _get_rouge_scorer():
    """Lazy loading for ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        try:
            _rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except Exception as e:
            print(f"Error loading ROUGE scorer: {e}")
            raise
    return _rouge_scorer

def _get_clip_model_and_processor():
    """Lazy loading for CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
    return _clip_model, _clip_processor

def extract_key_concepts(question: str) -> Dict[str, List[str]]:
    """Extract key emergency room medical concepts from the question.
    
    Args:
        question: The medical question about ER imagery
        
    Returns:
        Dictionary of key concepts organized by domain-specific categories
    """
    # Process with spaCy
    doc = nlp(question.lower())
    
    # Initialize results with ER-specific categories
    concepts = {
        'anatomical': [],
        'clinical_findings': [],
        'equipment': [],
        'procedure': [],
        'setting': [],
        'other_medical': []
    }
    
    # Extract entities recognized by spaCy
    for ent in doc.ents:
        if ent.label_ in ('DISEASE', 'ANATOMY', 'PROCEDURE'):
            if ent.label_ == 'DISEASE':
                concepts['clinical_findings'].append(ent.text)
            elif ent.label_ == 'ANATOMY':
                concepts['anatomical'].append(ent.text)
            elif ent.label_ == 'PROCEDURE':
                concepts['procedure'].append(ent.text)
    
    # Extract medical terms from our ER-specific custom lists
    for category, terms in MEDICAL_TERMS.items():
        for term in terms:
            # Check for exact matches and as part of phrases
            if any(re.search(fr'\b{re.escape(term)}\b', token.text.lower()) for token in doc):
                if term not in concepts[category]:
                    concepts[category].append(term)
    
    # Extract noun chunks that might be medical concepts
    for chunk in doc.noun_chunks:
        # Skip common non-medical terms
        if chunk.text.lower() in ['image', 'picture', 'photo']:
            continue
        
        # Check if this chunk contains a medical term from our categories
        found = False
        for category in concepts:
            if any(term in chunk.text.lower() for term in MEDICAL_TERMS.get(category, [])):
                found = True
                break
        
        # If not found in existing categories but seems medical, add to other_medical
        if not found and len(chunk.text) > 3:  # Skip very short terms
            concepts['other_medical'].append(chunk.text)
    
    # Clean up and deduplicate
    for category in concepts:
        concepts[category] = list(set(concepts[category]))
    
    return concepts


def identify_question_type(question: str) -> str:
    """Identify the type of emergency room medical question.
    
    Args:
        question: The medical question about ER imagery
        
    Returns:
        Question type specific to emergency room visual QA
    """
    question_lower = question.lower()
    
    # Descriptive questions (what is happening, what does it show) - prioritize this check first
    if any(keyword in question_lower for keyword in 
           ['what is', 'what are', 'what does', 'describe', 'explain', 'tell me about',
            'happening', 'shown', 'depicted', 'going on', 'activity', 'scene', 
            'what can you see', 'what do you see', 'showing', 'what appears']):
        return 'descriptive'
    
    # Procedural correctness questions (common in ER VQA)
    if any(keyword in question_lower for keyword in 
           ['proper', 'correct', 'appropriate', 'should', 'technique', 'protocol',
            'following', 'standard', 'adequately', 'adequacy', 'sufficient', 
            'properly', 'correctly', 'appropriately', 'adequate']):
        return 'procedural'
    
    # Patient status/condition questions
    if any(keyword in question_lower for keyword in 
           ['critical', 'stable', 'condition', 'status', 'sick', 'injured', 'ill',
            'severe', 'emergency', 'urgent', 'priority', 'triage']):
        return 'patient_status'
    
    # Equipment/facility questions
    if any(keyword in question_lower for keyword in 
           ['equipment', 'device', 'machine', 'monitor', 'setup', 'wearing', 'facility',
            'tools', 'instrument', 'system', 'apparatus', 'gear']):
        return 'equipment'
    
    # Location/setting questions
    if any(keyword in question_lower for keyword in 
           ['where', 'location', 'setting', 'room', 'area', 'facility', 'department', 
            'positioned', 'place', 'situated']):
        return 'setting'
    
    # Presence/identification questions
    if any(keyword in question_lower for keyword in 
           ['is there', 'can you see', 'do you observe', 'is it present', 'visible',
            'shown', 'depicted', 'appears', 'see', 'identify', 'recognize']):
        return 'identification'
    
    # Safety/PPE questions
    if any(keyword in question_lower for keyword in 
           ['safety', 'protection', 'ppe', 'gloves', 'mask', 'precaution', 'prevention',
            'sterile', 'clean', 'hygiene', 'contamination', 'exposure']):
        return 'safety'
    
    # Default to descriptive for general questions about the image
    if 'image' in question_lower or 'picture' in question_lower or 'photo' in question_lower:
        return 'descriptive'
    
    # Default type
    return 'general'


def get_traversal_strategy(question_type: str) -> str:
    """Determine the appropriate graph traversal strategy based on ER question type.
    
    Args:
        question_type: Type of the emergency room question
        
    Returns:
        Name of the traversal strategy to use
    """
    # Map ER-specific question types to appropriate traversal strategies
    er_strategy_map = {
        'procedural': 'procedure_evaluation',
        'patient_status': 'condition_assessment',
        'equipment': 'equipment_verification',
        'setting': 'location_identification',
        'identification': 'node_existence',
        'descriptive': 'reasoning_traversal',  # Use reasoning based traversal for descriptive queries
        'safety': 'safety_protocol_check',
        'general': 'reasoning_traversal'  # Changed default to reasoning traversal
    }
    
    # Fallback to generic strategies if ER-specific ones aren't implemented
    generic_strategy_map = {
        'procedural': 'subgraph_extraction',
        'patient_status': 'attribute_retrieval',
        'equipment': 'node_existence',
        'setting': 'attribute_retrieval',
        'identification': 'node_existence',
        'descriptive': 'reasoning_traversal',  # Use reasoning based traversal for descriptive queries
        'safety': 'subgraph_extraction',
        'general': 'reasoning_traversal'  # Changed default to reasoning traversal
    }
    
    # Use ER-specific strategy if available, otherwise fall back to generic
    return er_strategy_map.get(question_type, generic_strategy_map.get(question_type, 'reasoning_traversal'))


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference answer and hypothesis answer.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        BLEU score between 0 and 1
    """
    try:
        # Tokenize the sentences using a more robust approach
        # that doesn't rely on punkt_tab
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        
        # Create a list of references (just one in this case)
        references = [reference_tokens]
        
        # Use smoothing function to handle cases with no n-gram overlaps
        smoothie = SmoothingFunction().method1
        
        # Calculate BLEU score with weights emphasizing unigrams and bigrams
        # Since medical answers may have specialized terminology, unigrams are important
        weights = (0.5, 0.3, 0.1, 0.1)  # More weight to unigrams and bigrams
        
        bleu_score = sentence_bleu(references, hypothesis_tokens, 
                                  weights=weights, 
                                  smoothing_function=smoothie)
        return bleu_score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0


def calculate_multiple_bleu_scores(references: List[str], hypothesis: str) -> float:
    """Calculate BLEU score between multiple reference answers and a hypothesis answer.
    
    Args:
        references: List of reference texts (ground truths)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        BLEU score between 0 and 1, using the best matching reference
    """
    if not references:
        return 0.0
    
    # If only one reference, use the regular function
    if len(references) == 1:
        return calculate_bleu_score(references[0], hypothesis)
    
    try:
        # Tokenize the hypothesis using simple whitespace splitting
        # to avoid reliance on punkt_tab
        hypothesis_tokens = hypothesis.lower().split()
        
        # Tokenize all references
        tokenized_references = [ref.lower().split() for ref in references]
        
        # Use smoothing function to handle cases with no n-gram overlaps
        smoothie = SmoothingFunction().method1
        
        # Calculate BLEU score with weights emphasizing unigrams and bigrams
        weights = (0.5, 0.3, 0.1, 0.1)  # More weight to unigrams and bigrams
        
        # Calculate BLEU score using all references
        # NLTK's sentence_bleu automatically selects best matching n-grams from all references
        bleu_score = sentence_bleu(tokenized_references, hypothesis_tokens, 
                                  weights=weights, 
                                  smoothing_function=smoothie)
        return bleu_score
    except Exception as e:
        print(f"Error calculating multiple BLEU score: {e}")
        return 0.0


def normalize_text(text: str) -> List[str]:
    """Normalize text for BLEU score calculation.
    
    Args:
        text: Input text to normalize
        
    Returns:
        List of normalized tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize using simple whitespace splitting
    # to avoid reliance on punkt_tab
    tokens = text.split()
    
    return tokens


def calculate_rouge_score(reference: str, hypothesis: str) -> float:
    """Calculate ROUGE-L score between reference answer and hypothesis answer.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        ROUGE-L score between 0 and 1
    """
    try:
        scorer = _get_rouge_scorer()
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        return 0.0


def calculate_sentence_bert_similarity(reference: str, hypothesis: str) -> float:
    """Calculate SentenceBERT cosine similarity between reference and hypothesis.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        model = _get_sentence_transformer()
        
        # Encode the sentences
        reference_embedding = model.encode(reference, convert_to_tensor=True)
        hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(reference_embedding, hypothesis_embedding, dim=0)
        
        return cosine_similarity.item()
    except Exception as e:
        print(f"Error calculating SentenceBERT similarity: {e}")
        return 0.0


def calculate_entailment_score(reference: str, hypothesis: str) -> float:
    """Calculate entailment score for how well hypothesis is entailed by reference.
    
    Formula: ES = p(entailment|ref, gen)
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        Entailment probability score between 0 and 1
    """
    try:
        model, tokenizer = _get_nli_model_and_tokenizer()
        
        # Prepare input for the model (reference as premise, hypothesis as hypothesis)
        encoded_input = tokenizer(reference, hypothesis, return_tensors='pt', padding=True, truncation=True)
        
        # Get model prediction
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)
        
        # Return probability for entailment class (usually index 0)
        # Different models may have different indices for entailment, so check model documentation
        entailment_idx = 0  # Index for entailment in roberta-base-nli
        return probabilities[0, entailment_idx].item()
    except Exception as e:
        print(f"Error calculating entailment score: {e}")
        return 0.0


def calculate_clip_score(image_path: str, text: str) -> float:
    """Calculate CLIP score between an image and text.
    
    Args:
        image_path: Path to the image file
        text: Text to compare with the image
        
    Returns:
        CLIP score between 0 and 1
    """
    try:
        # Load CLIP model and processor
        model, processor = _get_clip_model_and_processor()
        
        # Load and preprocess the image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get similarity score
        logits_per_image = outputs.logits_per_image
        clip_score = logits_per_image[0].item()
        
        # Normalize to [0, 1] range
        clip_score = clip_score / 100.0
        
        return clip_score
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return 0.0


def calculate_clip_score_confidence(image_path: str, reference: str, hypothesis: str) -> float:
    """Calculate CLIP score confidence between image, reference, and hypothesis.
    
    Formula: CLIP-C = CLIP-S(img, gen) / (CLIP-S(img, ref) + CLIP-S(img, gen))
    
    Args:
        image_path: Path to the image file
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        
    Returns:
        CLIP score confidence between 0 and 1
    """
    try:
        # Calculate CLIP scores
        ref_score = calculate_clip_score(image_path, reference)
        gen_score = calculate_clip_score(image_path, hypothesis)
        
        # Calculate confidence
        denominator = ref_score + gen_score
        if denominator > 0:
            clip_confidence = gen_score / denominator
        else:
            clip_confidence = 0.0
            
        return clip_confidence
    except Exception as e:
        print(f"Error calculating CLIP score confidence: {e}")
        return 0.0


def calculate_all_metrics(reference: str, hypothesis: str, image_path: Optional[str] = None) -> Dict[str, float]:
    """Calculate all available metrics between reference and hypothesis.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (generated answer)
        image_path: Optional path to the image file for image-based metrics
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {}
    
    # Calculate text-based metrics
    metrics['bleu'] = calculate_bleu_score(reference, hypothesis)
    metrics['rouge'] = calculate_rouge_score(reference, hypothesis)
    metrics['sent'] = calculate_sentence_bert_similarity(reference, hypothesis) 
    metrics['ent'] = calculate_entailment_score(reference, hypothesis)
    
    # Calculate image-based metrics if image path is provided
    if image_path:
        try:
            metrics['clip_c'] = calculate_clip_score_confidence(image_path, reference, hypothesis)
        except Exception as e:
            print(f"Error calculating CLIP metrics: {e}")
            metrics['clip_c'] = 0.0
    
    return metrics 