# Repository for LLM intro session

## Sample 1: ChatGPT clone with Langchain and Streamlit

To demonstrate, how Langchain and Streamlit work together to give you visual access to an Azure OpenAI service instance, please review deep-dive-langchain.py

The script needs the following .env variables:

- AZURE_OAI_BASE_URL (your Azure OpenAI service deployment in the format of https://{YOUR-TENANT-HERE}.openai.azure.com)
- AZURE_OAI_KEY (your Azure OpenAI key)
- AZURE_OAI_DEPLOYMENTNAME (your Azure OpenAI deployment)

The file was inspired by: https://github.com/microsoft/az-oai-chatgpt-streamlit-harness