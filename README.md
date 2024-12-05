# **FR-NFR Classification Using Fine-Tuned Language Models**

This project involves training and fine-tuning state-of-the-art language models to classify requirements into **Functional Requirements (FR)** and **Non-Functional Requirements (NFR)**. Models were fine-tuned on two benchmark datasets, **PURE** and **PROMISE**, and evaluated using predictions on a ChatGPT-generated medical requirements dataset.

---

## **Models Trained**
The following models were fine-tuned for the FR-NFR classification task:

1. **BERT**: A bidirectional transformer pre-trained on a large corpus of English text.
2. **GPT-4o**: A lightweight and open-source version of GPT-4.
3. **RoBERTa**: An optimized version of BERT with better performance on NLP tasks.

---

## **Datasets**
### **Training Datasets**
1. **PURE Dataset**:
   - A dataset designed for requirements engineering tasks.
   - Includes annotated requirements labeled as FR and NFR.
   - https://fmt.isti.cnr.it/nlreqdataset/

2. **PROMISE Dataset**:
   - Another benchmark dataset in software requirements engineering.
   - Widely used for classification tasks in the requirements engineering domain.
   - Sonali, Sonali; Thamada,  Srinivasarao (2024), “FR_NFR_dataset”, Mendeley Data, V1, doi: 10.17632/4ysx9fyzv4.1

### **Evaluation Dataset**
- A **synthetic medical requirements dataset** generated using ChatGPT.
  - Consists of requirements relevant to the medical domain.
  - Labels (FR and NFR) were predicted using the fine-tuned models.

---

## **Process**
1. **Preprocessing**:
   - Cleaned and tokenized text data.
   - Mapped labels (`FR` and `NFR`) to numeric values (`0` and `1`).
   - Split datasets into training and testing subsets.

2. **Fine-Tuning**:
   - Fine-tuned BERT, GPT-4o, and RoBERTa using the PURE and PROMISE datasets.
   - Used the Hugging Face Transformers library for fine-tuning.

3. **Evaluation**:
   - Predictions were made on the ChatGPT-generated medical requirements dataset.
   - Labels (`0` → `FR`, `1` → `NFR`) were converted to human-readable format.
   - A classification report was generated, including precision, recall, and F1-score.

---

## **Results**
| Model     | Dataset  | **F1-Score (FR)** | **F1-Score (NFR)** |
|-----------|----------|-------------------|--------------------|
| BERT      | PURE     | **91%**           | **92%**            |
| BERT      | PROMISE  | **54%**           | **76%**            |
| RoBERTa   | PURE     | **93%**           | **94%**            |
| RoBERTa   | PROMISE  | **70%**           | **81%**            |
| GPT-4o    | PURE     | **96%**           | **96%**            |
| GPT-4o    | PROMISE  | **80%**           | **72%**            |


### **Inference**
To classify a new requirement:
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="fine_tuned_roberta", tokenizer="fine_tuned_roberta")
text = "The system shall allow up to 100 concurrent users."
result = classifier(text)
print(result)
```

---
