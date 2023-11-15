# Repository for LLM intro session

This repository should give attendees an overview of the LLM development ecosystem and should demonstrate how to solve basic use cases with the appropriate tools. The intention is to use the Azure OpenAI service as an LLM an to vary the different frameworks available to build usecases.
An overview about tools and frameworks could be found here:
![Overview of LLM development landscape from twitter.com](https://pbs.twimg.com/media/F9YsiGQboAISlLm?format=jpg&name=large)
(source: <https://twitter.com/chiefaioffice/status/1717614624793927972/photo/1>)

## Sample 1: ChatGPT clone with OpenAI and Streamlit

To demonstrate, how OpenAI and Streamlit work together to give you visual access to an Azure OpenAI service instance, please review deep-dive-openai.py

Within this solution Streamlit acts as a web server that generates a web frontend for the user. When the user inputs its text to the input textbox, this questions is sent to Azure OpenAI service through the Open AI Python SDK. The response and all previous messages were managed by the application to and sent as the history to the service.

![solution setup the ChatGPT clone](./pictures/1-openai.drawio.png)

The script needs the following .env variables:

- AZURE_OAI_BASE_URL (your Azure OpenAI service deployment in the format of https://{YOUR-TENANT-HERE}.openai.azure.com)
- AZURE_OAI_KEY (your Azure OpenAI key)
- AZURE_OAI_DEPLOYMENTNAME (your Azure OpenAI deployment)

To run the sample use:

```powershell
streamlit run 1-deep-dive-openai.py
```

The file was inspired by: https://github.com/microsoft/az-oai-chatgpt-streamlit-harness

## Sample 2: Chat with your PDF with Langchain and Streamlit

To enhace the first example, in this sample we add our own PDF knowledge to the chat with the GPT modell.

For this, we use a Langchain implementation for Retrieval Augmented Generation (RAG) with its ConversationalRetrievalChain, PyPDF to read the PDF file containing our knowledge base and FAISS vector store to store vectorized information about the PDF. Vectorization is done via chunking the whole document into single page chunks that were embedded during initialization with the OpenAI embedding modell and the resulting vectors were stored within the FAISS store (light gray lines). During answering the users questions, LangChains retrieval chain uses the user query, vectorizes this question and retrieves relevant passages of the PDF from FAISS. This relevant context is feed into the prompt to the GPT modell which than is able to answer based on the knowledge of the PDF (black lines).

![solution setup of the Langchain script](./pictures/2-langchain.drawio.png)

The script needs the following .env variables:

- AZURE_OAI_BASE_URL (your Azure OpenAI service deployment in the format of https://{YOUR-TENANT-HERE}.openai.azure.com)
- AZURE_OAI_KEY (your Azure OpenAI key)
- AZURE_OAI_DEPLOYMENTNAME (your Azure OpenAI deployment)
- AZURE_OAI_EMBEDDINGMODELL (your Azure OpenAI text embedding deployment)
- RAG_FILENAME (the path to your PDF file containing the knowledge for the RAG application)

To run the sample use:

```powershell
streamlit run 2-deep-dive-langchain-rag.py
```

The file was inspired by: https://github.com/yvann-ba/Robby-chatbot