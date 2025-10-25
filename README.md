# 🤖 AI Q&A Chatbot with LangChain

An intelligent question-answering chatbot powered by LangChain and HuggingFace embeddings for AI-related queries.

## ✨ Features

- **Semantic Search**: Uses sentence transformers for accurate answer retrieval
- **Dual Interface**: Both Streamlit and Flask implementations
- **Cosine Similarity**: Finds most relevant answers from knowledge base
- **Real-time Responses**: Instant answers to AI-related questions
- **Extensible Knowledge Base**: Easy to add new documents
- **Top-K Results**: Returns multiple relevant answers with confidence scores

## 🛠️ Tech Stack

- **Python 3.x**
- **LangChain** - LLM framework
- **HuggingFace Transformers** - Sentence embeddings
- **Streamlit** - Interactive web UI
- **Flask** - REST API backend
- **Scikit-learn** - Similarity calculations

## 📦 Installation

```bash
git clone https://github.com/vageeshgos/langchain_projects.git
cd langchain_projects
pip install -r requirements.txt
```

## 🚀 Usage

### Streamlit Interface
```bash
streamlit run app.py
```

### Flask API
```bash
python app.py
```

Access at `http://localhost:5000`

## 💡 How It Works

1. **Document Embedding**: Pre-computes embeddings for knowledge base
2. **Query Processing**: Converts user questions to embeddings
3. **Similarity Search**: Finds top-3 most relevant documents
4. **Response Generation**: Returns answers with confidence scores

## 📚 Knowledge Base Topics

- Artificial Intelligence fundamentals
- Machine Learning concepts
- Deep Learning architectures
- Natural Language Processing
- Computer Vision
- Popular AI models (ChatGPT, Gemini, Claude)
- Transformers and neural networks
- AI agents and automation

## 🔧 Customization

Add new documents to the knowledge base:
```python
documents = [
    "Your new AI fact or concept here",
    # Add more documents
]
```

## 📊 Model Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Similarity Metric**: Cosine similarity
- **Top-K Results**: 3 most relevant answers

## 🤝 Contributing

Pull requests welcome! Enhance the knowledge base or improve the UI.

## 👤 Author

**Vageesh Goswami**
- GitHub: [@vageeshgos](https://github.com/vageeshgos)
- LinkedIn: [vageesh-goswami](https://www.linkedin.com/in/vageesh-goswami/)
- Portfolio: [vageesh-goswami-portfolio.base44.app](https://vageesh-goswami-portfolio.base44.app/)
