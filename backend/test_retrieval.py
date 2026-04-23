import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def test_retrieval():
    load_dotenv()
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )
        print(f"Index: {index_name}")
        query = "what is digicoco"
        print(f"Testing retrieval for: {query}")
        results = vectorstore.similarity_search(query, k=1)
        if results:
            print(f"Success! Found {len(results)} results.")
            print(f"Content snippet: {results[0].page_content[:100]}...")
        else:
            print("No results found.")
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    test_retrieval()
