import streamlit as st
from streamlit_chat import message
import dotenv
import os
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

ENV = dotenv.dotenv_values(".env")

#OAI config
api_key = ENV['AZURE_OAI_KEY']
api_base = ENV['AZURE_OAI_BASE_URL']
api_type = "azure"
api_version = "2023-05-15"
deployment_name = ENV["AZURE_OAI_DEPLOYMENTNAME"] 
embedding_name = ENV["AZURE_OAI_EMBEDDINGMODELL"]

os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
os.environ["OPENAI_API_BASE"] = api_base 
os.environ["OPENAI_API_VERSION"] = api_version

llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    openai_api_version=api_version,
    model_name="gpt-35-turbo"
)

embed_model = OpenAIEmbeddings(model='text-embedding-ada-002',
                              deployment=embedding_name,
                              openai_api_base=api_base,
                              openai_api_type='azure',
                              openai_api_key=api_key,
                              chunk_size=1)


#load docs
def load_docs(file):
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    print(f'Loaded {len(docs)} documents')
    vectors = FAISS.from_documents(docs, embed_model)
    chain = ConversationalRetrievalChain.from_llm(
        llm = AzureChatOpenAI(
            temperature=0.0,
            model_name='gpt-3.5-turbo', 
            deployment_name=deployment_name,
            openai_api_base=api_base,
            openai_api_type=api_type,
            openai_api_key=api_key,
            openai_api_version=api_version),
        retriever=vectors.as_retriever())
    return chain

#Streamlit config
st.set_page_config(page_title="My own ChatGPT with Streamlit and Langchain")
st.title("My own ChatGPT with Streamlit and Langchain")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#Communicate with OpenAI model
def generate_response(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['messages']})
    st.session_state['messages'].append((prompt, result["answer"]))    
    return result["answer"]

#load the sample doc
chain = load_docs(ENV['RAG_FILENAME'])

#Streamlit UI & action stuff
response_container = st.container()
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        print(f'User input {user_input}')
        output = generate_response(
            user_input
        )
        print(f'GPT response {output}')
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="shapes",
            )
            message(
                st.session_state["generated"][i], key=str(i), avatar_style="identicon"
            )