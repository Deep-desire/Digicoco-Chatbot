import os
from dotenv import load_dotenv
from supabase import create_client
import inspect

load_dotenv()
url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client = create_client(url, key)
print(inspect.signature(client.table("users").upsert))
