import os

from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI

# Api key 
os.environ["OPENAI_API_KEY"] = "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"

# Loading docs
loader = TextLoader("docs/doc1.txt", encoding='utf-8')
documents = loader.load()
documents += TextLoader("docs/doc2.txt", encoding='utf-8').load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# vector store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)

# similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# using openapu
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# Setting up Rag 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# running the chatbot
print(" RAG Chatbot Ready! Type 'exit' to quit.")
while True:
    query = input("You: ") 
    if query.lower() == "exit":
        break
    result = qa_chain({"query": query})
    print("\nBot:", result["result"])
