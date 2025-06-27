import os 

# Step 1: Setup LLM (Mistral with huggingFace)
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# Load environment token and repo
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN) 

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )

# Step 3: Load the vector database
FAISS_DB_PATH = "vector_db/faiss_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create the custom chain
def run_custom_chain(user_query: str):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=user_query)

    response = client.chat_completion(messages=[
        {"role": "user", "content": formatted_prompt}
    ])

    return {
        "result": response.choices[0].message["content"],
        "source_documents": docs
    }

# Step 5: Query the custom chain
user_query = input("Enter your question: ")

response = run_custom_chain(user_query)
print("Answer:", response['result'])

print("\nSource Documents:\n")
for i, doc in enumerate(response["source_documents"], start=1):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "unknown")
    print(f"Document {i} — Source: {source}, Page: {page}\nContent: {doc.page_content[:300]}...\n")
