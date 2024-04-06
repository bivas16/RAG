# RAG.py

import warnings
import textwrap
import time
from langchain_community.llms import LlamaCpp
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Config:
    # Configuration parameters here...
    model_name = 'llama2-7b'
    temperature = 0
    top_p = 0.95
    repetition_penalty = 1.15
    split_chunk_size = 800
    split_overlap = 0
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'
    k = 3
    PDFs_path = 'File_data'
    Embeddings_path = 'lit_embedding'

def initialize_llm():
    # Initialize and return LlamaCpp instance
    llm = LlamaCpp(
        model_path="/Users/bivasbisht/Thesis/llama-2-7b-chat.Q5_K_M.gguf",
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        verbose=True,
    )
    return llm

def initialize_embeddings():
    # Initialize and return HuggingFaceInstructEmbeddings instance
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=Config.embeddings_model_repo
    )
    return embeddings

def load_vector_db_embeddings(embeddings):
    # Load vector DB embeddings
    vectordb = FAISS.load_local(
        Config.Embeddings_path,
        embeddings
    )
    return vectordb

def initialize_prompt_template():
    # Initialize and return PromptTemplate instance
    prompt_template = """
    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return PROMPT

def initialize_retriever(vectordb):
    # Initialize and return retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": Config.k, "search_type": "similarity"})
    return retriever

def initialize_qa_chain(llm, retriever, prompt_template):
    # Initialize and return RetrievalQA instance
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        verbose=False
    )
    return qa_chain

def wrap_text_preserve_newlines(text, width=700):
    # Wrap text while preserving newlines
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    # Process LLM response
    ans = wrap_text_preserve_newlines(llm_response['result'])
    sources_used = ' \n'.join([
        source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
        for source in llm_response['source_documents']
    ])
    ans = ans + '\n\nSources: \n' + sources_used
    return ans 

def llm_ans(qa_chain, query):
    # Perform QA with LLM and return response
    start = time.time()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str

def main():
    # Main function to execute the code and get user query input
    warnings.filterwarnings("ignore")

    llm = initialize_llm()
    embeddings = initialize_embeddings()
    vectordb = load_vector_db_embeddings(embeddings)
    prompt_template = initialize_prompt_template()
    retriever = initialize_retriever(vectordb)
    qa_chain = initialize_qa_chain(llm, retriever, prompt_template)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = llm_ans(qa_chain, query)
        print(answer)

if __name__ == "__main__":
    main()
