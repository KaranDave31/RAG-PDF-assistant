#Basic Flow
#Load PDF
#Text Splitting
#Embedding
#Store in vector database
#Retrieval
#RAG Response

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

#Loading PDF
loader = PyPDFLoader('BIVA NOTES.pdf')
docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[1].metadata)

#Text Splitting
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separator=''
)

result = splitter.split_documents(docs)

#print(result[100].page_content)

#Embedding
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(result, embedding_model)

def retrieve_docs(query, k=5):
    docs_faiss = faiss_db.similarity_search(query, k=k)
    #return docs_faiss
    return [doc.page_content for doc in docs_faiss]

query = "Explain pie chart"
retrieve_docs_results = retrieve_docs(query)

# for idx, content in enumerate(retrieve_docs_results, 1):
#     print(f"{idx}. {content}\n")


chat_model = ChatMistralAI(
    model_name="mistral-large-latest",
    temperature=0.7
)

def ask_llm(query, k=5):
    docs = retrieve_docs(query, k)
    context = "\n\n".join(docs)  

    messages = [
        SystemMessage(content="You are a research assistant "
                              "and must answers in the context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return chat_model.invoke(messages).content

print(ask_llm("Explain pie chart"))