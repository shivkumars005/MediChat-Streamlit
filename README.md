# MediBot: Medical Question Answering ChatBot

MediBot is an intelligent chatbot built with **LangChain**, **Hugging Face Inference API**, **FAISS**, and **Streamlit** to answer medical questions based on a custom PDF knowledge base (e.g. `GALE_ENCYCLOPEDIA---.pdf`). It retrieves the most relevant context from your documents and answers accurately using the **Mistral-7B-Instruct-v0.3** model.

---

## LIVE
The app is now live on streamlit and you can view it from [here](https://medibot-by-sks.streamlit.app/)

## 🚀 Features

* RAG pipeline using FAISS for semantic search
* Uses `sentence-transformers/all-MiniLM-L6-v2` for embedding
* Runs **Mistral 7B** via Hugging Face Inference API (`conversational` task)
* Streamlit chat UI with conversation memory
* Shows top source documents with page and preview content

---

## 📁 Project Structure

```
├── medibotUI.py                  #Streamlitapp
├── connect_llm.py                # CLI version
├── create_memory.py              # CLI version
├── vector_db/faiss_db            # Vector index
    ├──index.faiss
    ├──index.pkl  
├── data/GALE_ENCYCLOPEDIA.pdf    # Source PDF
├── requirements.txt              # Python dependencies
└── README.md                     # This file
├──LICENSE                        # License
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/shivkumars005/MediChat-Streamlit
cd MediChat-Streamlit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Export Hugging Face API Token

```bash
export HF_TOKEN=your_huggingface_token  # or set in .env
```

### 4. Build Vector Index (one-time)

Use this script to create `vector_db/faiss_db` from PDF(s).

### 5. Run the App

```bash
streamlit run medibot_UI.py
```

---

## 🧠 How It Works

1. User enters a question in Streamlit.
2. FAISS finds top-k relevant document chunks from the PDF.
3. Prompt is constructed with retrieved context.
4. Mistral LLM via Hugging Face chat API responds.
5. Response + sources are shown in UI.

---

## 📚 Requirements

* Python 
* Streamlit
* LangChain
* HuggingFace Hub
* FAISS

---

## ✅ Credits

* [LangChain](https://github.com/langchain-ai/langchain)
* [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [Streamlit](https://streamlit.io)

---

## 🔒 License

MIT License

---

## 📞 Collaboration

* [LinkedIn](https://linkedin.com/in/shivakumarsouta)
* [Portfolio](https://shivakumarsouta-portfolio.vercel.app/)
* [EMail](mailto:shivakumarsouta18@gmail.com)
