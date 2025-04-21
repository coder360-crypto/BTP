# Focused Concept Graph VQA with MI-Based Hallucination Control

A framework for medical visual question answering that improves accuracy and reliability by reducing hallucinations through focused analysis, knowledge graph representations, and mechanistic interpretability checks.

## Overview

This project implements a novel approach to Medical VQA that does not require model training, instead focusing on structured analysis and verification:

1. **Region of Interest Detection**: Identifies the most relevant regions in medical images using a log-probability increase method
2. **Focused Analysis**: Guides the VLM's attention to relevant regions by masking/blurring non-ROI areas
3. **Concept Extraction**: Extracts structured medical concepts and relationships from the focused image using NVIDIA's Gemma 3-27B-IT model
4. **Visual Grounding Check**: Verifies the extracted concepts using mechanistic interpretability principles
5. **Knowledge Graph Construction**: Builds a reliable graph representation with only well-grounded information
6. **Intelligent Graph Traversal**: Uses various strategies based on question type
7. **Hallucination-Aware Answer Synthesis**: Generates answers using only verified graph information

## Installation

### Prerequisites

- Python 3.8+
- `uv` for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medical-vqa.git
   cd medical-vqa
   ```

2. Create and activate a virtual environment using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys (see `.env.example` for template)
   - The project uses NVIDIA's API with Gemma 3-27B-IT model by default
   - Default API key is provided in `.env.example`

5. Download required models (if not automatically downloaded):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Command Line Interface

The system can be used to process individual images or batch process multiple images from a CSV file:

#### Single Image Mode

```bash
python main.py --image ervqa_data/test_v2_images/headinjuriesinhospital3.jpeg --question "Is the patient's condition critical?" --output-dir ./outputs
```

#### Batch Processing Mode

```bash
python main.py --csv ervqa_data/test_vqa_v4.csv --images-dir ervqa_data/test_v2_images/ --output-dir ./outputs
```

### Testing the Pipeline

You can test the pipeline functionality with:

```bash
# Run with mock responses (no API calls)
python test_pipeline.py --mock

# Run with actual NVIDIA API calls
python test_pipeline.py
```

## Architecture

The system is composed of several core components:

### Pipeline Phases

1. **Question Analysis**
   - Extract key medical concepts
   - Identify question type
   - Select appropriate graph traversal strategy

2. **ROI Detection**
   - Calculate log probability changes with region perturbation
   - Identify regions that significantly affect the answer

3. **Image Focusing**
   - Apply Gaussian blur to non-ROI regions
   - Create a focused image that emphasizes relevant areas

4. **Concept Extraction**
   - Use NVIDIA's Gemma 3-27B-IT model to extract structured medical concepts
   - Output anatomical structures, findings, attributes, and relationships

5. **Grounding Check**
   - Apply mechanistic interpretability principles
   - Calculate grounding scores for each extracted concept

6. **Knowledge Graph Construction**
   - Filter out low-grounded information
   - Build a reliable medical knowledge graph

7. **Graph Traversal & Answer Synthesis**
   - Apply specialized traversal strategies based on question type
   - Generate answers using only verified information

### Directory Structure

```
├── src/
│   ├── core/            # Core pipeline components
│   │   ├── roi_detector.py
│   │   ├── concept_extractor.py
│   │   ├── answer_synthesizer.py
│   │   └── vqa_pipeline.py
│   ├── graphs/          # Knowledge graph implementation
│   │   └── knowledge_graph.py
│   ├── utils/           # Utility functions
│   │   ├── image_utils.py
│   │   └── nlp_utils.py
│   ├── visualization/   # Visualization modules
│   └── data/            # Data handling modules
├── main.py              # Main entry point
├── test_pipeline.py     # Test script
├── requirements.txt     # Dependencies
└── .env.example         # Template for environment variables
```

## Configuration

The system is configurable through the `.env` file or command-line arguments:

- **VLM Settings**
  - `VLM_API_KEY`: NVIDIA API key for Gemma 3-27B-IT model
  - `VLM_MODEL`: Model to use (default: google/gemma-3-27b-it)
  - `VLM_TEMP`: Temperature for model generation

- **ROI Detection**
  - `ROI_DETECTION_THRESHOLD`: Threshold for considering a region important
  - `BLUR_SIGMA`: Blur intensity for non-ROI regions

- **Graph Parameters**
  - `MIN_GROUNDING_SCORE`: Minimum score to include concepts in graph
  - `NODE_SIMILARITY_THRESHOLD`: Threshold for considering nodes semantically similar

## Extending the System

The modular design allows for easy extension:

- Add new VLM integrations in `concept_extractor.py`
- Implement additional graph traversal strategies in `knowledge_graph.py`
- Create custom visualization modules in the `visualization/` directory

## License

[MIT License](LICENSE)

## Acknowledgments

This project is based on concepts from research in medical VQA, mechanistic interpretability, and knowledge graphs. Specific papers that inspired this approach include:

- Log-Probability Increase methods for understanding VLM behavior
- Mechanistic Interpretability techniques for detecting hallucinations
- Knowledge graph approaches to structured reasoning 

# Medical VQA Test Pipeline

This repository contains a test pipeline for the Medical Visual Question Answering system.

## Setup

1. Install dependencies using uv:

```bash
uv pip install -r requirements.txt
```

2. Download required NLTK resources:

```bash
python setup_nltk.py
```

3. Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Running Tests Without ROI Detection and Grounding

To run the pipeline tests without ROI detection and without grounding:

```bash
python test_pipeline.py --disable-roi-detection
```

The script will process the test images and questions, and generate the following outputs in the `./test_outputs` directory:

- Individual results for each sample in `sample_X` subdirectories
- Aggregated metrics in `metrics_results.csv`
- Consolidated results in `outputs.json`
- Summary statistics in `summary.json`
- Visualizations of metric distributions in the `graphs` subdirectory:
  - Individual histograms for each metric (BLEU, ROUGE, SentenceBERT, Entailment, CLIP)
  - Box plots comparing all metrics
  - Bar charts showing mean, median, and mode for each metric

## Additional Options

- `--mock`: Run in mock mode without making actual API calls
- `--csv PATH`: Specify a custom CSV file with test data (default: `ervqa_data/test_vqa_v4.csv`)
- `--images-dir PATH`: Specify the directory containing test images (default: `ervqa_data/test_v2_images/`)
- `--limit N`: Limit the number of samples to process (default: 5)
- `--output-dir PATH`: Specify the output directory (default: `./test_outputs`)

## Metrics Explained

The evaluation metrics used in this pipeline include:

- **BLEU-1 Score**: Measures n-gram precision, with emphasis on unigrams
- **ROUGE-L Score**: Measures the longest common subsequence between generated and reference answers
- **SentenceBERT Similarity**: Measures semantic similarity using BERT embeddings
- **Entailment Score**: Measures how well the generated answer is entailed by the reference answer
- **CLIP-Score Confidence**: Measures how well the generated answer aligns with the image, normalized by reference answer alignment 