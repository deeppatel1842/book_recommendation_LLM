# 📚 Semantic Book Recommendation with LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A smart book recommendation system that understands the semantic meaning and emotional context of your input using Large Language Models (LLMs).

![Demo Screenshot](images/demo_screenshot.png)

## 🔧 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
bash
git clone git@github.com:deeppatel1842/book_recommendation_LLM.git
cd book_recommendation_LLM

Install dependencies

bash
pip install -r requirements.txt

💡 Features
Semantic understanding of book descriptions

Emotion-aware recommendations

Interactive Gradio interface

🧠 Technical Architecture
Core Components
Semantic Search
sentence-transformers/all-MiniLM-L6-v2 for text embeddings

Sentiment Analysis
j-hartmann/emotion-english-distilroberta-base for emotional context extraction

Data Flow
User input processing

Embedding generation

Similarity matching

Emotion-based filtering

Recommendation generation

🚀 Usage
Running the Application
bash
python gradio_dashboard.py

