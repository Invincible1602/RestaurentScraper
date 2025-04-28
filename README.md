# RestaurentScraper & RAG Chatbot

A Python‐based tool and Streamlit application to scrape restaurant data, build a vector index, and interact with a Retrieval‑Augmented Generation (RAG) chatbot powered by FAISS and a Hugging Face LLM.

---

## 🚀 Project Overview

**RestaurentScraper** automates gathering restaurant information from web sources and exports structured CSV/JSON outputs. The **Restaurant RAG Chatbot** leverages that data by embedding restaurant descriptions into a FAISS index and querying a Hugging Face model for conversational answers.


---

## 📦 Installation & Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Invincible1602/RestaurentScraper.git
   cd RestaurentScraper
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # macOS/Linux
   .\.venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**  
   Copy `.env.example` to `.env` and set:
   ```dotenv
   HUGGINGFACE_API_KEY=your_hf_api_key_here
   ```
   The app also reads these internal flags:
   ```bash
   export TRANSFORMERS_NO_TF=1
   export USE_TF=0
   ```

5. **Build the FAISS index**  
   Run your scraper to generate `csv_vector_index.faiss` and `csv_metadata.json` in the project root:
   ```bash
   python main1.py 
   ```

---

Adjust the **RAG** parameters at the top of `main4.py` (or wherever you import):

```python
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
INDEX_PATH = "csv_vector_index.faiss"
DATA_PATH = "csv_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELEVANCE_THRESHOLD = 10.0
TOP_K = 5
```

---

## 🏃 Usage

### 1. Launch the Chatbot

```bash
streamlit run main4.py
```  
Then open http://localhost:8501 in your browser.

---

## 🔍 How It Works

1. **Data Ingestion**: `main1.py` uses Selenium or HTTP requests + BeautifulSoup to scrape restaurant details into CSV/JSON.
2. **Vector Indexing**: Text descriptions are embedded with **all-MiniLM-L6-v2** and stored in a FAISS `IndexFlatL2` (see `scraper.py`).
3. **Streamlit App**:
   - Loads embedding model (`SentenceTransformer`) and FAISS index via `@st.cache_resource`.
   - On user query:
     - Compute query embedding
     - Perform nearest‑neighbor search
     - Filter by **RELEVANCE_THRESHOLD** and dietary keywords (e.g. hide `vegetarian` for non‑veg queries)
     - Format context and send a prompt to Hugging Face inference API
     - Display the generated answer
4. **Error Handling**:
   - Missing index/metadata: shows a Streamlit error banner
   - Missing HF API key: returns an error message
   - API failures: logged and surfaced to the user

---

## 🔧 Customization

- **Change LLM**: Update `HF_API_URL` to another model endpoint.
- **Thresholds & K**: Tweak `RELEVANCE_THRESHOLD` and `TOP_K` for broader/narrower context.
- **Embedding Model**: Swap to any compatible SentenceTransformer (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
- **UI Layout**: Modify `streamlit` calls (e.g. add images, tables).

---

## 🤝 Contributing

1. Fork & clone
2. Create a feature branch 
3. Commit & push
4. Open a Pull Request

Follow code style and add tests for new features.

---

## Screenshot

<img width="1470" alt="Screenshot 2025-04-28 at 4 56 41 PM" src="https://github.com/user-attachments/assets/024c8803-8b20-4531-aefe-724ff4b2292a" />



## 📜 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

