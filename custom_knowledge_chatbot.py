from flask import Flask, request, jsonify
from pathlib import Path
from typing import List, Tuple
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain

# Constants
local_path = 'orca-mini-3b.ggmlv3.q4_0.bin' 
model_path = 'ggml-all-MiniLM-L6-v2-f16.bin' 
text_path = "./docs/state_of_the_union.txt"
index_path = "./full_sotu_index"

# Functions
def initialize_embeddings() -> GPT4AllEmbeddings:
    return GPT4AllEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: GPT4AllEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path, n_threads=4, f16_kv=True, use_mlock=True, embedding=True)

embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# vectorstore = generate_index(chunks, embeddings)
# vectorstore.save_local("full_sotu_index")

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=2048)

app = Flask(__name__)
###Unity chatbot response
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    chat_history = []
    result = qa({"question": prompt, "chat_history": chat_history})
    return jsonify({'output': result['answer']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# # # # Chatbot loop
# chat_history = []
# print("Welcome to the State of the Union chatbot! Type 'exit' to stop.")
# while True:
#     query = input("Please enter your question: ")
    
#     if query.lower() == 'exit':
#         break
#     result = qa({"question": query, "chat_history": chat_history})

#     print("Answer:", result['answer'])
   