import main
main._init_supabase_client()
print('supabase client is', 'set' if getattr(main,'_supabase_client',None) else 'None')
import os
print('NEXT_PUBLIC_SUPABASE_URL=', repr(os.environ.get('NEXT_PUBLIC_SUPABASE_URL')))
print('SUPABASE keys loaded:', 'SUPABASE_SERVICE_ROLE_KEY' in os.environ, 'NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY' in os.environ)
