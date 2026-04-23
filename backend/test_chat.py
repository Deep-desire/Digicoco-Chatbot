import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_chat():
    load_dotenv()
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    
    try:
        print(f"Testing chat with deployment: {deployment}")
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print("Success!")
        print(f"Reply: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Chat failed: {e}")

if __name__ == "__main__":
    test_chat()
