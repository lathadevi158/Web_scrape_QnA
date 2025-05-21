import streamlit as st
import langchain, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

langchain.debug=True

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    model_name=os.getenv("AZURE_MODEL_NAME"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    temperature=0
)

# Define the scraping function
def scrape(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    bs_transformer = Html2TextTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["title", "p", "strong", "h3", "span"])

    # Split the documents
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    
    return splits

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("AZURE_API_KEY"),
    deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    openai_api_type="azure",
    chunk_size=1
)
# Create the custom prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    """
)

def main():
    st.title("Question Answering System")

    # Input fields
    url = st.text_input("Enter URL")
    question = st.text_area("Enter your question")

    if st.button("Get Answer"):
        # Scrape and process the documents
        extracted_content = scrape(url)
        split_chunks = []
        for doc in extracted_content:
            page_content = doc.page_content
            chunks = text_splitter.split_text(page_content)

            for chunk in chunks:
                split_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

        # Create Chroma vector store
        db = Chroma.from_documents(split_chunks, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        # Create a retrieval-based QA system
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            retriever=retriever
        )

        # Get the answer to the query
        result = qa({"query": question})

        # Extract the answer from the result
        answer = result.get('result', 'Information not available')

        # Display the answer
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()