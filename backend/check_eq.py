import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client = create_client(url, key)
try:
    client.table("sessions").select("*").filter("session_id", "eq", "test_id").execute()
    print("Filter succeeded!")
except Exception as e:
    print(vars(e))
