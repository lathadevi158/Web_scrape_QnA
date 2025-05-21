# 🤖 Web Scrape Question Answering System with LangChain & Azure OpenAI

This project enables you to **ask questions about any webpage content by providing its URL**. The system scrapes the webpage, processes the text, and answers your questions in **concise, natural language**, powered by **LangChain**, **Azure OpenAI**, and **Streamlit**.

---

## 🚀 Features

- 🌐 Input any URL to scrape content  
- ✂️ Automatically split large text into chunks for efficient processing  
- 🧠 Use Azure OpenAI embeddings to create a searchable vector database  
- 🔍 Retrieve relevant context chunks using similarity search  
- 💬 Get clear, concise answers (max 3 sentences) to your questions  
- 🖥️ Easy-to-use Streamlit interface

---

## 🧰 Tech Stack

| Tool                  | Purpose                               |
|-----------------------|-------------------------------------|
| Streamlit             | Frontend UI                         |
| LangChain             | Document loading, text splitting, QA chaining |
| Azure OpenAI          | Embeddings & Chat-based LLM         |
| Chroma                | Vector database for document search |
| python-dotenv         | Environment variable management      |

---

## 📁 Project Structure

.
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── .env # Environment variables for Azure keys
├── README.md # Project documentation


---

## 🖥️ How It Works

1. Enter a URL in the input box.  
2. The app scrapes and extracts text content from the page.  
3. The text is split into chunks and embedded using Azure OpenAI embeddings.  
4. Chunks are stored in a Chroma vector store for similarity search.  
5. When you ask a question, the most relevant chunks are retrieved.  
6. The Azure ChatOpenAI model answers your question based on the retrieved context.  
7. You get a concise and relevant answer displayed on the screen.

---

## ▶️ Run the App

streamlit run app.py
Open your browser at http://localhost:8501 (Streamlit usually opens it automatically).

📄 License
This project is licensed under the MIT License.

🙋‍♀️ Author
Marpally Latha Devi,
Prompt Engineer | Generative AI Developer,
GitHub: lathadevi158
