# 📚 Scalable Academic Policy QA System  
### (Big Data Retrieval using TF-IDF, MinHash LSH & SimHash)

---

## 👨‍💻 Overview

This project is a **scalable Question Answering (QA) retrieval system** built over university academic handbooks (UG/PG).  
Instead of relying on a chatbot-style system, the focus is on **efficient information retrieval at scale** using Big Data techniques.

The system retrieves relevant document chunks using multiple retrieval strategies and compares their performance in terms of **accuracy, speed, and scalability**.

---

## 🎯 Objectives

- Efficient retrieval over large academic policy documents  
- Implementation of **approximate similarity techniques**
  - MinHash + LSH  
  - SimHash  
- Baseline implementation using **TF-IDF + cosine similarity**
- Comparative analysis of:
  - Accuracy  
  - Query latency  
  - Scalability tradeoffs  

---

## 🏗️ System Architecture

                +----------------------+
                |   PDF Handbook       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Text Preprocessing |
                | - Cleaning          |
                | - Chunking (200–500)|
                +----------+-----------+
                           |
         +-----------------+------------------+
         |                 |                  |
         v                 v                  v
+----------------+  +----------------+  +----------------+
| TF-IDF Index   |  | MinHash + LSH  |  |   SimHash     |
| (Exact Match)  |  | (Approximate)  |  | (Fingerprint) |
+----------------+  +----------------+  +----------------+
         \                 |                 /
          \                |                /
           \               v               /
            +---------------------------+
            |   Query Processing Layer |
            +---------------------------+
                           |
                           v
                +----------------------+
                | Top-K Retrieved Text |
                +----------------------+


---

## ⚙️ Retrieval Methods

### 1. TF-IDF (Baseline)

- Uses vector space model
- Computes cosine similarity between query and document chunks

**Strengths:**
- High precision for keyword-based queries  
- Strong baseline for evaluation  

**Limitations:**
- Poor handling of semantic similarity  
- Weak on paraphrased queries  

---

### 2. MinHash + LSH (Approximate Retrieval)

- Converts text into **shingles**
- Generates MinHash signatures
- Uses LSH buckets for fast similarity search

**Strengths:**
- Efficient for large-scale datasets  
- Reduces comparison complexity significantly  

**Limitations:**
- Sensitive to threshold and shingle size  
- May return no matches for short queries  

**Enhancement Added:**
- Fallback to MinHash Jaccard similarity when LSH returns empty results  

---

### 3. SimHash (Fingerprint-Based)

- Converts text into binary fingerprints  
- Uses **Hamming distance** for similarity comparison  

**Strengths:**
- Extremely fast  
- Good for approximate matching  

**Limitations:**
- Lower precision for semantic retrieval  
- Can return irrelevant chunks  

---

## 🔄 Query Processing Flow

1. User enters query  
2. Query is processed through:
   - TF-IDF retrieval  
   - LSH retrieval (with fallback)  
   - SimHash retrieval  
3. Top-K relevant chunks are displayed with:
   - text snippet  
   - score or distance  

---

## 📊 Experimental Observations

### TF-IDF
- Best performance for exact keyword queries  
- Highly accurate for structured policy text  

---

### LSH
- Fast retrieval but sensitive to parameters  
- Sometimes fails under strict thresholds  
- Fallback mechanism improves robustness  

---

### SimHash
- Always returns results  
- Fast but less accurate  
- Useful for approximate similarity detection  

---

## ⚖️ Tradeoff Analysis

| Method   | Accuracy | Speed | Scalability |
|----------|----------|-------|--------------|
| TF-IDF   | High     | Medium| Low          |
| LSH      | Medium   | High  | High         |
| SimHash  | Low-Med  | Very High | Very High |

---

## 🧠 Key Insights

- Exact methods are more accurate but less scalable  
- Approximate methods improve scalability at the cost of precision  
- LSH performance depends heavily on parameter tuning  
- Hybrid systems are necessary for real-world retrieval tasks  

---


---

## 🚀 How to Run

### 1. Setup environment

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 1. Setup environment

python src/preprocess.py

### 3. Run System

python src/main.py
