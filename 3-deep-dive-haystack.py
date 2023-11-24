import gradio as gr
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter
from haystack.nodes import PreProcessor
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from haystack.nodes import AnswerParser
from haystack.nodes.prompt import PromptNode, PromptModel, PromptTemplate
from haystack.pipelines import Pipeline
import dotenv

#Toggle to review RAG search results and scores
DEBUG_RAG = False

ENV = dotenv.dotenv_values(".env")
pipeline = Pipeline()

#load the RAG document
def load_documents():
    #reads PDF
    converter = PDFToTextConverter(
        remove_numeric_tables=True,
        valid_languages=["de","en"]
    )
    docs = converter.convert(file_path=ENV['RAG_FILENAME'], meta=None)

    #do the chunking
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=512,
        split_overlap=32,
        split_respect_sentence_boundary=True,
    )

    docs_to_index = preprocessor.process(docs)
    return docs_to_index

#create the haystack pipeline
def init_pipeline(docs_to_index):
    #Create document store in RAM 
    document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)
    
    #Vector similarity retriever with Huggingface model
    dense_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=True,
        scale_score=False,
    )
    
    document_store.delete_documents()
    document_store.write_documents(docs_to_index)
    document_store.update_embeddings(retriever=dense_retriever)
    
    #Word similarity retriever with standard BM25
    sparse_retriever = BM25Retriever(document_store=document_store)

    #Promptmodel to use Azure OpenAI service
    promptmodel_azureopenai = PromptModel(
        model_name_or_path="gpt-35-turbo-16k",
        api_key=ENV["AZURE_OAI_KEY"],
        model_kwargs={
            "azure_base_url": ENV["AZURE_OAI_BASE_URL"],
            "azure_deployment_name": ENV["AZURE_OAI_DEPLOYMENTNAME"],
        }
    )

    rag_prompt = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text for the given question.
                                Provide a clear and concise response that summarizes the key points and information presented in the text.
                                Your answer should be in your own words and be no longer than 50 words.
                                \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
        output_parser=AnswerParser(),
    )

    #Node to communicate with Azure OpenAI LLM
    prompt_node = PromptNode(
        model_name_or_path=promptmodel_azureopenai,
        default_prompt_template=rag_prompt,
        api_key=ENV["AZURE_OAI_KEY"],
        max_length=16000
    )
    
    #node to join the documents from dense and sparse retrieval
    join_documents = JoinDocuments(join_mode="concatenate")

    #node to adjust the ranking order of the joined documents by using a Huggingface model
    rerank = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

    #finally assemble the pipeline
    pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
    pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
    pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])

    if not DEBUG_RAG == True:
        pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["ReRanker"])

#callback handler for Gradio chat interface
def get_response(message, history):
    if DEBUG_RAG == True:
        prediction = pipeline.run(
            query=message,
            params={
                "SparseRetriever": {"top_k": 10},
                "DenseRetriever": {"top_k": 10},
                "JoinDocuments": {"top_k_join": 15, "debug":True},
                "ReRanker": {"top_k": 5},
            },
        )
    else:
        prediction = pipeline.run(
            query=message,
            params={
                "SparseRetriever": {"top_k": 10},
                "DenseRetriever": {"top_k": 10},
                "JoinDocuments": {"top_k_join": 15},
                "ReRanker": {"top_k": 5},
            },
        )

    if DEBUG_RAG == True:
        for doc in prediction["documents"]:
            print(f"Score {doc.score} \t Content: {doc.content}")
            print("\n-------------------------------------------\n")
    else:
        return prediction["answers"][0].answer

#everything needed to create the Gradio chatinterface
iface = gr.ChatInterface(get_response)

#main routing to start the application
if __name__ == "__main__":
    docs = load_documents()
    init_pipeline(docs)
    iface.launch()