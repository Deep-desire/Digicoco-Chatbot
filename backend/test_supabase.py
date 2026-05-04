import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
client = create_client(url, key)

resp = client.table("users").select("*").limit(5).execute()
print("users:", resp.data)

resp = client.table("messages").select("*").limit(5).execute()
print("messages:", resp.data)
