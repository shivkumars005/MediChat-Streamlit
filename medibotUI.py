import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient

@st.cache_resource
def load_vector_db(FAISS_DB_PATH):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )

def main():
    st.set_page_config(page_title="MediBot", page_icon="Images/favicon.ico", layout="wide")
    st.title("Ask MediBot!")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your question here:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """
        CUSTOM_PROMPT_TEMPLATE = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vector_store = load_vector_db("vector_db/faiss_db")
            if vector_store is None:
                st.error("Failed to load the vector database.")

            client = InferenceClient(model=HUGGING_FACE_REPO_ID, token=HF_TOKEN)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)
            formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=prompt)

            response = client.chat_completion(messages=[
                {"role": "user", "content": formatted_prompt}
            ])
            response_content = response.choices[0].message["content"]

            st.chat_message("assistant").markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            with st.expander("Show source documents"):
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "unknown")
                    st.markdown(f"**Document {i}** — Source: `{source}`, Page: `{page}`\n\n> {doc.page_content[:300]}...")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "An error occurred while processing your request."})

if __name__ == "__main__":
    main()