from typing import List
from datetime import datetime
from langchain.document_loaders import TextLoader
from langchain.llms import CTransformers
from langchain.embeddings import GPT4AllEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#CTransformers configuration
config = {
    'top_k': 40, #This is used for token prediction determination/ the lower the value the higher the output quality and the lower diversity; the higher the value the lower the quality and higher diversity.
    'top_p': 0.95, # This is used for token prediction determination similar to top_k/ additionally, it randomly selects the tokens that are in the top (default.95) threshold
    'temperature': 0.1, # This adds what i would consider a prediction weight to the tokens/ the closer to 0 the more sure the model is of its predictions which leads to a higher selection of the top tokens. The closer to 1.0 the less certain and hence the more evenly distributed the probabilites giving the chance of a less likely token to be selected
    'repetition_penalty': 1.3, # A multiplier for penalizing repetative tokens more than one means higher penalty, lower than one means less penalty and more repitition
    'last_n_tokens': 64, # Gives a window of tokens for the model to look back at and and determine if it is a repeated token, if it is then the penalty gets applied
    'seed': -1, # Used to initialize the random number generator for the model. This number set to -1 means the seed will be randomly chosen which means the random number range will be random and in turn you will get unpredictable responses. For more reproducable responses change the seed number to be anything above zero.
    'max_new_tokens': 256, # This sets the length of the response you will get back from the model
    'stop': None, # makes the model stop based on a customized string argument. for instance if you want to make the model stop at the end of a paragraph you might set stop to ['\n']
    'stream': False, # Used to stream the tokens since we are using Langchain this will be accomplished through a callback
    'reset': True, # Resets the model state before generating text,  Resetting the model state means clearing this internal state so that it can be reinitialized with new data. This is useful when you want to start processing a new sequence of data with a fresh internal state12
    'batch_size': 8, # Typically used for training/evaluating, tt batches the feed(or input) into managable chunks, the higher the number the faster the speed but the more resources it will need
    'threads': -1, # Sets the number of threads used for parallel execution of tasks and can speed up computations, typically set to the number of cores available on your CPU/ setting the value to -1 means the system should automatically choose the number of threads to use.
    'context_length': -1, # This sets the length of the context window the llm will use when generating the next token, -1 means that it will decide automatically how much it will use. 
    'gpu_layers': 0, # To be used for offloading to the GPU, Most libraries that are used for quantization of the llm for gpu offloading are better supported on linux/ Additionally, this can be used to share the offlaoding bewtweent he CPU and GPU (very cool concept, I NEED LINUX)
}

llm = CTransformers(model='C:\LocalLLMProject\models\llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama', callbacks=[StreamingStdOutCallbackHandler()], config=config)
model_path='ggml-all-MiniLM-L6-v2-f16.bin' 
text_path = "./docs/true-history-of-the-world.txt"
index_path = "./history-of-the-world-index"

# Functions
def initialize_embeddings() -> GPT4AllEmbeddings:
    return GPT4AllEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: GPT4AllEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def similarity_search(query, index):
    matched_docs = index.similarity_search(query, k=4)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# vectorstore = generate_index(chunks, embeddings)
# vectorstore.save_local("history-of-the-world-index")

index = FAISS.load_local(index_path, embeddings)


### Vector Store Memory
embedding_size = 1536 # Dimensions of the Embeddings
memory_index = faiss.IndexFlatL2(embedding_size)
embedding_fn = GPT4AllEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, memory_index, InMemoryDocstore({}), {})

memory_retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=memory_retriever)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """

You are an urban gangster, your responses are spoken as a gangster.

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.You must answer in its original language.

{context}

Question: {question}
Answer:"""


QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT, memory=memory, verbose=True)
## chatbot style query
qa = ConversationalRetrievalChain(
    retriever=index.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
)

chat_history=[]

query = "Where did the seeds come from?"
result = qa({"question": query, "chat_history": chat_history})

query = "what do you think of the seeds?"
result = qa({"question": query, "chat_history": chat_history})






 # # Context based query
# question = "How did magic come to be?"
# matched_docs, sources = similarity_search(question, index)

# template = """
# You are a urban gangster who only answers questions based off of the context provided.\ 
# If you do not know an answer to a question, then respond with you do not know.\
# Please use the following context to answer questions.\
# Context: {context}
# ---
# Question: {question}
# """

# context = "\n".join([doc.page_content for doc in matched_docs])
# prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# llm_chain.run(question)
