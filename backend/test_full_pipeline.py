import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_azure_openai_client():
    return AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

def get_vectorstore():
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )
    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
    )

def test_stream_pipeline():
    query = "what is digicoco"
    print(f"Testing stream pipeline for: {query}")
    
    try:
        # 1. Retrieval
        vs = get_vectorstore()
        matches = vs.similarity_search_with_relevance_scores(query, k=5)
        context = "\n\n".join([m[0].page_content[:2000] for m in matches])
        print(f"Retrieved context length: {len(context)}")
        
        # 2. Stream
        client = get_azure_openai_client()
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        system_prompt = "Context: {context}" # Simplified for test
        
        print("Starting stream...")
        stream = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt.replace("{context}", context)},
                {"role": "user", "content": query},
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)
        print("\nStream finished successfully.")
        
    except Exception as e:
        print(f"\nPipeline failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_stream_pipeline()
