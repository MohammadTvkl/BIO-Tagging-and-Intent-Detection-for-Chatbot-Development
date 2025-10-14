# BIO-Tagging and Intent Detection for Chatbot Development

### 🧠 Overview
This project focuses on **BIO-Tagging** and **Intent Detection** for developing an intelligent Persian chatbot that assists users in mobile phone selection.  
The work was conducted as part of my academic internship at the University of Isfahan’s *Creative Industries Research Center* (Summer 2024).

The goal was to implement and compare deep learning models — from a custom **Bi-GRU** to transformer-based **ParsBERT** — to determine which approach performs best for Persian Natural Language Understanding (NLU) tasks.

---

### 🚀 Project Objectives
- Implement **BIO-Tagging** for entity recognition (slot filling).
- - Training a **Bi-GRU** model
- Perform **Intent Detection** to classify user goals.
- Compare **custom-trained GRU** and **pre-trained ParsBERT** models.
- Evaluate models by **F1-score, accuracy, inference speed, and RAM usage**.
- Identify the best trade-off between performance and efficiency for chatbot deployment.

---

### 🧩 Models Implemented

| Model | Framework | Description | Strengths | Weaknesses |
|--------|------------|-------------|------------|-------------|
| **Bi-GRU** | TensorFlow/Keras | Custom sequence labeling model with Embedding, Bidirectional GRU, and TimeDistributed layers. | Fast inference, low RAM usage | Slightly lower F1-score |
| **ParsBERT (uncased-fa)** | Hugging Face Transformers | Fine-tuned Persian BERT model for token classification. | High accuracy | Higher inference time |
| **Multilingual BERT** | Hugging Face Transformers | cased multilingual version trained on 100+ languages. | Works cross-lingually | Heavy memory footprint |
| **Base-zwnj-fa-BERT** | HooshvareLab | Persian-tuned BERT model for token tagging. | Well-suited for Persian text | Slightly slower on GPU |

---

### 📚 Dataset
- ~13,000 Persian sentences.
- Each record includes:  
  **Sentence**, **BIO Tags**, and **Intent Label**.
- Example:
Sentence: یه گوشی سامسونگ بخر
Tags: O B-Product B-Brand O
Intent: suggest_mobile


The dataset was tokenized, padded, and encoded for model input using TensorFlow and Hugging Face tokenizers.

---

### ⚙️ Implementation Details
#### Bi-GRU Pipeline
- Tokenization and padding of sentences.
- BIO tag encoding (B, I, O + PAD).
- Model architecture:  
- `Embedding` (128 dim)  
- `Bidirectional(GRU, 128 units)`  
- `TimeDistributed(Dense, softmax)`
- Optimizer: `Adam`, Loss: `Categorical Crossentropy`, Metric: `Accuracy`.

#### ParsBERT Pipeline
- Pretrained model: `HooshvareLab/bert-base-parsbert-uncased`
- Fine-tuned for token classification using:
- `TFBertForTokenClassification`
- `TF_USE_LEGACY_KERAS=1` for library compatibility
- Evaluation metrics: precision, recall, F1-score
- Benchmarked RAM and time per prediction using `psutil` and `time`.

---

### 📊 Results Summary

| Model | Avg. F1-Score | Avg. Time per Sentence (s) | RAM Usage (GB) |
|--------|----------------|----------------------------|----------------|
| **Bi-GRU** | ~0.91 | **0.08** | **0.30** |
| **ParsBERT (uncased-fa)** | ~0.94 | 0.20 | 0.35 |
| **Multilingual BERT** | ~0.93 | 0.24 | 0.57 |
| **base-zwnj-fa-BERT** | ~0.93 | 0.23 | 0.53 |

➡️ **Conclusion:**  
- GRU is lightweight and faster — ideal for resource-limited chatbot deployments.  
- ParsBERT offers higher accuracy — suitable when precision is more critical than speed.

---

### 🧪 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Hugging Face Transformers**
- **ParsBERT**
- **pandas**, **NumPy**, **scikit-learn**
- **Google Colab (GPU)**
- **psutil** for memory profiling

📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- RAM Consumption
- Inference Time

