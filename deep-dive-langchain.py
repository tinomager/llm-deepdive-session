import streamlit as st
from streamlit_chat import message
import dotenv
import openai

ENV = dotenv.dotenv_values(".env")

#OAI config
openai.api_version = "2023-05-15"
openai.api_base = ENV['AZURE_OAI_BASE_URL']
openai.api_key = ENV['AZURE_OAI_KEY']
openai.api_type = "azure"

system_message = 'Du bist ein Assistent der Leuten hilft.'

#Streamlit config
st.set_page_config(page_title="My own ChatGPT with Streamlit and Langchain")
st.title("My own ChatGPT with Streamlit and Langchain")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_message}]

#Communicate with OpenAI model
def generate_response(prompt):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    try:
        completion = openai.ChatCompletion.create(
            engine=ENV["AZURE_OAI_DEPLOYMENTNAME"],
            messages=st.session_state["messages"],
            
        )
        response = completion.choices[0].message.content
    except openai.error.APIError as e:
        st.write(response)
        response = f"The API could not handle this content: {str(e)}"
    st.session_state["messages"].append({"role": "assistant", "content": response})
    return response

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