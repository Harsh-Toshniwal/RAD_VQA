# Model Output Analysis - Question Type Classifier

## Overview
The question type classifier predicts whether a medical VQA question is **OPEN** or **CLOSED**.

## Model Architecture
- Input: Question text (tokenized and embedded)
- Model: Question Classifier with:
  - Word embeddings (GloVe 300d)
  - GRU-based question encoder (1024d)
  - Attention mechanism
  - FC layers: 1024 → 256 → 64 → **2** (final output)
- Output: **2 raw scores (logits)** - one for each class

## Output Format

### Raw Model Output
The model outputs 2 numerical scores (logits):
- `output[0]`: Score for **OPEN** class
- `output[1]`: Score for **CLOSED** class

**Prediction:** The class with the **higher score** is selected.

### Example Outputs

#### Example 1 - Correctly Predicted OPEN Question
```
Question: "is there evidence of an aortic aneurysm"
Ground Truth: OPEN
Prediction: OPEN ✓

Model Output Scores:
  OPEN:   11.2924  ← Higher score (OPEN predicted)
  CLOSED: -26.7398
```
The model is highly confident this is an OPEN question (large positive score for OPEN, large negative for CLOSED).

#### Example 2 - Correctly Predicted CLOSED Question
```
Question: "how is the patient oriented"
Ground Truth: CLOSED
Prediction: CLOSED ✓

Model Output Scores:
  OPEN:   -6.8382
  CLOSED:  8.1568  ← Higher score (CLOSED predicted)
```
The model correctly predicts CLOSED (CLOSED score is positive and higher than OPEN).

#### Example 3 - MISCLASSIFIED (Error Case)
```
Question: "what do the two small hyperintensities indicate"
Ground Truth: OPEN
Prediction: CLOSED ✗

Model Output Scores:
  OPEN:   -7.2946
  CLOSED:  7.7930  ← Higher score (CLOSED predicted incorrectly)
```
The model incorrectly predicts CLOSED when the ground truth is OPEN.

## Interpretation Guide

### Understanding the Scores
1. **Logits (Raw Scores):** The output values are unnormalized scores
   - Positive values indicate the model "leans toward" that class
   - Negative values indicate the model "leans away" from that class
   - The absolute magnitude indicates confidence

2. **Prediction:** `argmax(output)` = class with highest score
   - If `output[0] > output[1]`: Predict OPEN
   - If `output[1] > output[0]`: Predict CLOSED

3. **Confidence:** The difference between scores indicates confidence
   - Large difference (e.g., 11.29 vs -26.74): Very confident
   - Small difference (e.g., 5.36 vs -12.74): Less confident but still clear

### Converting to Probabilities (Optional)
If you want probabilities instead of logits, apply softmax:
```python
probabilities = torch.softmax(output, dim=1)
# Example: [11.29, -26.74] → [1.0000, 0.0000]  (100% OPEN, 0% CLOSED)
```

## Test Set Results

### Overall Performance
- **Accuracy:** 99.33% (448/451 correct)
- **Errors:** 3 misclassifications

### Label Distribution
From the test set samples shown:
- Majority class: OPEN questions
- Minority class: CLOSED questions

### Error Cases (3 mistakes out of 451)

**Error 1:**
```
Question: "what do the two small hyperintensities indicate"
Ground Truth: OPEN
Prediction: CLOSED
Scores: OPEN=-7.29, CLOSED=7.79
```

**Error 2:**
```
Question: "is the spleen present"
Ground Truth: CLOSED
Prediction: OPEN
Scores: OPEN=5.36, CLOSED=-12.08
```

**Error 3:**
```
Question: "the small hypo-dense of air seen in the lumen normal or"
Ground Truth: OPEN
Prediction: CLOSED
Scores: OPEN=-4.88, CLOSED=5.03
```

## Question Type Definitions

### OPEN Questions
Questions requiring descriptive answers (not just yes/no):
- "what is the abnormality?"
- "where is the fracture located?"
- "what organ is this?"

### CLOSED Questions
Questions with yes/no or specific short answers:
- "is there a fracture?"
- "is the heart enlarged?"
- "how is the patient oriented?" (has fixed answer options)

## Dataset Details
- **Test Set:** 451 medical VQA questions from VQA-RAD dataset
- **Source:** Medical imaging questions about radiology images
- **Note:** Some questions contain `<UNK>` tokens (unknown words not in vocabulary)

## Files Generated
- `detailed_predictions.json`: Contains all 451 predictions with:
  - Original questions
  - Ground truth labels
  - Predicted labels
  - Raw output scores for both classes
  - Confidence values
  - Correctness flags

## Usage
To see detailed predictions for any number of samples:
```bash
python eval_detailed.py --show_samples 50
```

To check the saved predictions:
```python
import json
with open('detailed_predictions.json', 'r') as f:
    results = json.load(f)
    
# Access predictions
for pred in results['predictions'][:10]:
    print(f"Question: {pred['question']}")
    print(f"GT: {pred['ground_truth']}, Pred: {pred['prediction']}")
    print(f"Scores: {pred['output_scores']}")
    print()
```
