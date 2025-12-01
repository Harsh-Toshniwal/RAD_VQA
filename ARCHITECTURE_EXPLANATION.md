# RAD VQA Model Architecture Explanation

## Overview
The RAD VQA (Radiology Visual Question Answering) model is a **multi-modal deep learning system** that combines visual and textual information to answer questions about medical (radiology) images. It uses a combination of **Bilinear Attention Networks (BAN)**, pre-trained feature extractors, and separate classifiers for open/closed-ended questions.

---

## High-Level Architecture Flow

```
Input
├─ Medical Image (RGB or grayscale)
├─ Question (text)
└─ Answer Type (OPEN or CLOSED)
        │
        ├──────────────────────────────────────────────────────────────┐
        │                                                              │
        ▼                                                              │
VISUAL FEATURE EXTRACTION                                             │
├─ MAML (Model-Agnostic Meta-Learning)                               │
│   └─ Input: 84×84 image → 4-layer CNN → Output: 64-dim vector     │
│                                                                    │
├─ Auto-Encoder (Denoising Auto-Encoder)                            │
│   └─ Input: 128×128 image                                         │
│      ├─ Encoder: Conv1(64) → MaxPool → Conv2(32) → MaxPool →     │
│      │           Conv3(16)                                        │
│      └─ Output: 16 feature maps (used as encoder features)        │
│                                                                    │
└─ Visual Embedding: [64-dim MAML + 64-dim AE] = 128-dim (or just   │
                                                  one if disabled)    │
        │                                                              │
        ├──────────────────────────────────────────────────────────────┤
        │                                                              │
        ▼                                                              │
QUESTION PROCESSING                                                  │
├─ Word Embedding Layer                                             │
│   └─ Input: question token IDs → Embedding lookup → 300-dim      │
│      (initialized with GloVe embeddings)                           │
│                                                                    │
├─ Question Embedding (GRU)                                         │
│   └─ 300-dim word embeddings → Bidirectional GRU → 1024-dim      │
│      output for each word position (12 tokens max)                │
│                                                                    │
└─ Question Classification (separate classifier)                    │
    └─ Question vector → FC layers (1024 → 256 → 64 → 2 classes)   │
       Output: CLOSED (0) or OPEN (1) question type                 │
        │                                                              │
        └──────────────────────────────────────────────────────────────┤
                                                                      │
CORE BAN (Bilinear Attention Network)                                │
                                                                      │
├─ Bilinear Attention (BiAttention)                                 │
│   ├─ Input: visual_features [batch, 1, 128] + question_emb [batch, 12, 1024]
│   ├─ BCNet (Bilinear Connect Network)                             │
│   │   └─ Projects: V → [1, 1024×3], Q → [12, 1024×3]             │
│   │   └─ Computes: attention logits [batch, glimpse=2, 1, 12]     │
│   ├─ Softmax over question positions                              │
│   └─ Output: attention weights [batch, 2, 1, 12]                  │
│                                                                    │
├─ BiResNet (Bilinear Residual Network)                             │
│   └─ Loops 2 times (glimpse=2):                                   │
│      ├─ BCNet.forward_with_weights(): Uses attention weights to   │
│      │  reweight visual-question interactions                      │
│      ├─ Projects output through FC layers [hid_dim → hid_dim]     │
│      ├─ Adds residual connection (updated question)               │
│      └─ Output dimension stays 1024                               │
│                                                                    │
├─ Type Attention (multiplies by question type features)            │
│   └─ Separate attention weights based on question type           │
│                                                                    │
└─ Result: Fused representation [batch, 1024]                       │
                                                                    │
        ▼                                                              │
SEPARATE CLASSIFICATION HEADS                                        │
                                                                    │
├─ CLOSED-ENDED Questions:                                          │
│   └─ SimpleClassifier: [1024 → 2048 → 56 classes]                │
│      (56 possible closed answers)                                 │
│                                                                    │
├─ OPEN-ENDED Questions:                                            │
│   └─ SimpleClassifier: [1024 → 2048 → 431 classes]               │
│      (431 possible open answers)                                  │
│                                                                    │
└─ Total Answer Space: 56 + 431 = 487 possible answers             │
        │                                                              │
        ▼                                                              │
OUTPUT                                                                │
                                                                    │
├─ Logits [batch, num_candidates]                                  │
├─ Argmax → answer class index                                      │
├─ +56 offset for OPEN answers (to map to global label2ans)        │
└─ Label lookup → final answer text                                 │
```

---

## Component Breakdown

### 1. **Visual Feature Extraction**

#### MAML (Model-Agnostic Meta-Learning) - `maml.py`
- **Purpose**: Extract features from medical images using pre-trained CNN
- **Input**: 84×84 grayscale image
- **Architecture**: 
  - Conv2d(1, 64, k=3, s=2, p=1) + BatchNorm + ReLU
  - Conv2d(64, 64, k=3, s=2, p=1) + BatchNorm + ReLU
  - Conv2d(64, 64, k=3, s=2, p=1) + BatchNorm + ReLU
  - Conv2d(64, 64, k=3, s=2, p=1) + BatchNorm + ReLU
- **Output**: 64-dimensional feature vector (average pooled)
- **Training**: Frozen (pre-trained weights loaded)

#### Auto-Encoder (Denoising Auto-Encoder) - `auto_encoder.py`
- **Purpose**: Learn low-level image representations from noisy inputs
- **Input**: 128×128 medical image
- **Encoder**:
  - Conv2d(1, 64, k=3, p=1) + ReLU + MaxPool2d(2)
  - Conv2d(64, 32, k=3, p=1) + ReLU + MaxPool2d(2)
  - Conv2d(32, 16, k=3, p=1) + ReLU
- **Decoder** (reconstruction for training):
  - ConvTranspose2d(16, 32) + Conv2d(32, 32) + ReLU
  - ConvTranspose2d(32, 64) + Conv2d(64, 1) + Sigmoid
- **Usage**: Encoder output is converted to 64-dim vector via Linear(16384 → 64)
- **Output**: 64-dimensional feature vector

#### Combined Visual Representation
- If both MAML and Auto-Encoder enabled: **[64 + 64] = 128-dim**
- If only one: **64-dim**

---

### 2. **Question Processing**

#### Word Embedding - `language_model.py`
- **Input**: Token IDs (0 = padding, 1-N = vocabulary)
- **Layer**: nn.Embedding(vocab_size, 300)
- **Output**: 300-dimensional word vectors (12 tokens × 300-dim)
- **Initialization**: Pre-trained GloVe 6B embeddings loaded

#### Question Embedding (RNN Encoder) - `language_model.py`
- **Input**: 300-dim word embeddings [batch, 12, 300]
- **Architecture**: 
  - Bidirectional GRU(300 → 1024, num_layers=1)
  - Returns full sequence output (not just final state)
- **Output**: [batch, 12, 1024] — sequence of question word representations

---

### 3. **Question Type Classifier** - `classify_question.py`

Predicts whether a question is **OPEN-ENDED** (1) or **CLOSED-ENDED** (0).

- **Input**: Question token IDs
- **Processing**:
  1. Word embedding (300-dim) + Question GRU (1024-dim)
  2. Question Attention mechanism (attends to relevant question words)
  3. FC layers: 1024 → 256 (ReLU) → 64 (Dropout, ReLU) → **2 (logits for CLOSED/OPEN)**
- **Output**: 2-class logits
- **Usage**: Determines which answer classifier head to use

---

### 4. **Bilinear Attention Network (BAN)** - `model.py` / `multi_level_model.py`

The **core reasoning module** that fuses visual and textual information.

#### BiAttention Module
- **Purpose**: Learn attention over questions conditioned on visual features
- **Components** (`connect.py`):
  - **FCNet**: General-purpose fully connected network with weight normalization
  - **BCNet** (Bilinear Connect Network): Bilinear interaction between visual and question features
  
- **Forward Pass**:
  ```
  Visual Features: [batch, 1, 128]
  Question Embeddings: [batch, 12, 1024]
  
  v_net: 128 → 1024*k (project visual)
  q_net: 1024 → 1024*k (project question)
  
  h_mat: learnable glimpse matrix [1, glimpse=2, 1, 1024*k]
  
  logits = (v_features * h_mat) @ q_features.T  [batch, 2, 1, 12]
  attention = softmax(logits)  [batch, 2, 1, 12]
  ```
  
  - **Output**: Attention weights [batch, glimpse=2, num_objects=1, question_len=12]
  - **Interpretation**: For each "glimpse" (2 different attention heads), how much weight each question word gets

#### BiResNet (Bilinear Residual Network)
- **Purpose**: Iteratively refine the question representation using visual information
- **Loop**: Runs for `glimpse` iterations (2 times)
  
- **Each iteration**:
  ```
  1. BCNet.forward_with_weights():
     - Takes: visual features, question embeddings, attention weights
     - Computes: weighted bilinear interaction v^T * W * q
     - Output: [batch, 1024]
  
  2. Project through FC: [1024 → 1024]
  
  3. Residual connection: new_q = FC(bilinear_output) + old_q
  ```

- **Final Output**: Sum of all glimpse outputs → **[batch, 1024]** fused representation

#### Type Attention (Additional refinement)
- **Purpose**: Modulate the fused representation based on question type
- **Approach**: Uses same word embedding + question GRU + FC layers to produce scaling vector
- **Operation**: Final output = BiResNet output * type_attention_vector

---

### 5. **Answer Classification Heads** - `classifier.py`

Two separate `SimpleClassifier` instances (one for OPEN, one for CLOSED).

#### SimpleClassifier Architecture
```
Input: [batch, 1024] (fused representation)
  ↓
Weight-normalized Linear: 1024 → 2048
  ↓
ReLU activation
  ↓
Dropout(0.5)
  ↓
Weight-normalized Linear: 2048 → num_candidates
  ↓
Output logits: [batch, num_candidates]
```

- **For CLOSED questions**: num_candidates = 56
- **For OPEN questions**: num_candidates = 431
- **Output**: Raw logits (not probabilities)

---

### 6. **Data Processing & Label Mapping** - `dataset_RAD.py`

#### Label Organization
```
label2ans (all 487 answers):
├─ Index 0-55:    56 CLOSED answers
└─ Index 56-486:  431 OPEN answers

label2close (56 answers)
label2open (431 answers)
```

#### Training Target Construction
- For a CLOSED question sample:
  ```
  composed_target = zeros(487)
  composed_target[0:56] = target_distribution_over_close_answers
  ```
  
- For an OPEN question sample:
  ```
  composed_target = zeros(487)
  composed_target[56:487] = target_distribution_over_open_answers
  ```

- **Target format**: Soft targets (confidence scores) or binary (sparse, one-hot)

#### Prediction Time (Inference)
1. Question classifier outputs type (OPEN/CLOSED)
2. Select relevant model head (close_classifier or open_classifier)
3. Argmax over logits → index
4. If OPEN: add offset +56 to convert local index to global label index
5. Look up text: `answer_text = label2ans[global_index]`

---

## Training Flow

1. **Input**: Batch of (image, question, answer)
2. **Forward Pass**:
   - Visual extraction (MAML + AutoEncoder)
   - Question embedding (Word emb + GRU)
   - BiAttention + BiResNet (fused representation)
   - Classify question type
   - Route to appropriate answer classifier (OPEN or CLOSED)
3. **Loss Computation**:
   - If AutoEncoder enabled: reconstruction loss + classification loss
   - Main: CrossEntropyLoss between logits and target distribution
4. **Backward Pass**: Update all parameters (except frozen MAML)

---

## Key Design Choices

| Component | Why? |
|-----------|------|
| **Bilinear Attention** | Captures fine-grained interactions between visual and textual modalities |
| **Multiple Glimpses** | Multiple passes allow iterative refinement of question understanding |
| **Separate OPEN/CLOSED heads** | Different answer spaces; better specialization per question type |
| **Pre-trained MAML** | Captures low-level medical image features via meta-learning |
| **Auto-Encoder** | Learns compact image representations; can also be used for reconstruction |
| **Type Attention** | Question type affects how visual/textual info should be fused |
| **Weight Normalization** | Stabilizes training in bilinear modules |

---

## Summary Statistics

| Parameter | Value |
|-----------|-------|
| Vocabulary size | ~3,000 tokens |
| Question max length | 12 tokens |
| Word embedding dim | 300 |
| Question GRU hidden dim | 1024 |
| Visual feature dim (MAML) | 64 |
| Visual feature dim (AutoEncoder) | 64 |
| Total visual dim | 128 (or 64 if one disabled) |
| Bilinear hidden dim | 1024 |
| Number of glimpses (attention heads) | 2 |
| Closed answer candidates | 56 |
| Open answer candidates | 431 |
| **Total possible answers** | **487** |
| Classifier hidden dim | 2048 |

---

## Differences from Standard VQA

1. **Medical domain**: Uses pre-trained MAML for medical image understanding
2. **Open/Closed split**: Not typical in standard VQA; reflects medical question nature
3. **Type attention**: Additional refinement based on question type classification
4. **Denoising auto-encoder**: Extra visual encoding pathway (not standard)
5. **TF-IDF weighted embeddings** (optional): Can re-weight word embeddings by importance

---

## References

- **Bilinear Attention Networks (BAN)**: Jin-Hwa Kim et al., "Bilinear Attention Networks" (ICLR 2018)
- **MAML**: Model-Agnostic Meta-Learning for visual feature extraction
- **Auto-Encoder**: Denoising auto-encoder for medical image feature learning
- **Dataset**: RAD (Radiology Question Answering Dataset)
