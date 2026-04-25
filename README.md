# 📚 Scalable Academic Policy QA System

## 📌 Overview

This project implements a **scalable Question-Answering (QA) system** over university academic handbooks.
The goal is to design an efficient **retrieval pipeline** that can handle large text data and return relevant policy information based on user queries.

> ⚠️ Note: This is **not a chatbot**. The system focuses on **retrieval**, where answers are grounded in actual handbook content.

---

## 🚀 Day 1 Progress (Baseline System)

The following components have been implemented:

### ✅ Data Ingestion & Preprocessing

* Extracted text from PDF using `pdfplumber`
* Cleaned text (lowercasing, removing noise)
* Split content into fixed-size chunks (~300 words)
* Stored chunks in `chunks.json`

### ✅ Baseline Retrieval System

* Implemented **TF-IDF + Cosine Similarity**
* Indexed all chunks for fast retrieval
* Returns **top-k most relevant chunks** for a query

### ✅ Query Interface

* Simple CLI-based interface
* Displays:

  * Relevance score
  * Chunk ID
  * Preview of retrieved text

### ✅ Engineering Practices

* Virtual environment (`venv`) for isolation
* Dependency management via `requirements.txt`
* Clean project structure
* Incremental Git commits
* `.gitignore` to exclude unnecessary files

---

## 🧠 System Pipeline (Current)

```
PDF Handbook
     ↓
Text Extraction
     ↓
Cleaning
     ↓
Chunking (~300 words)
     ↓
TF-IDF Indexing
     ↓
User Query
     ↓
Cosine Similarity
     ↓
Top-k Relevant Chunks
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/YOUR_USERNAME/academic-qa-system.git
cd academic-qa-system
```

---

### 2. Create Virtual Environment

```
python -m venv venv
```

---

### 3. Activate Environment (Windows)

```
venv\Scripts\activate
```

---

### 4. Install Dependencies

```
pip install -r requirements.txt
```

---

### 5. Run Preprocessing

```
python src/preprocess.py
```

---

### 6. Run QA System

```
python src/main.py
```

---

## 🧪 Example Queries

### ✅ Works Well (Keyword-based)

* `minimum gpa requirement`
* `attendance policy`
* `course withdrawal policy`
* `repeat course policy`

### ❌ Not Suitable (Semantic / Opinion-based)

* `what is nust`
* `convince me to join nust`
* `is nust a good university`

---

## ⚠️ Known Limitations (Day 1)

* TF-IDF relies on **exact keyword matching**
* Does not understand semantic meaning
* Sensitive to:

  * Synonyms (`counseling` vs `counselling`)
  * Natural language queries
* May return irrelevant chunks for vague questions

> These limitations are intentional and will be addressed in later stages using **approximate similarity techniques (LSH, SimHash)**.

---

## 📊 Observations

* Performs well for **policy-related factual queries**
* Struggles with **open-ended or conversational queries**
* Highlights the gap between **exact vs semantic retrieval**

---

## 🔜 Next Steps (Planned)

* Implement **MinHash + LSH** for approximate similarity
* Implement **SimHash** with Hamming distance
* Compare:

  * Accuracy
  * Query time
  * Scalability
* Add evaluation metrics (Precision@k, latency)
* Optional: ranking improvements (PageRank-style scoring)

---

## 📁 Project Structure

```
academic-qa-system/
│
├── data/
│   └── handbook.pdf
│
├── src/
│   ├── preprocess.py
│   ├── tfidf_retrieval.py
│   ├── main.py
│
├── chunks.json
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🧾 Reproducibility

This project is fully reproducible:

```
pip install -r requirements.txt
```

All dependencies are version-locked to ensure consistent results.

---

## 🎯 Key Takeaway

This baseline demonstrates that:

> **Exact retrieval methods (TF-IDF) are simple and effective for keyword queries, but insufficient for scalable, semantic search systems.**

This motivates the need for **approximate similarity techniques**, which will be implemented in subsequent stages.

---
