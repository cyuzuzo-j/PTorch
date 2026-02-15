from sqlalchemy import create_engine, pool
try:
    print("Attempting to create engine with pool arguments...")
    engine = create_engine('sqlite:///:memory:', pool_size=5, max_overflow=10)
    print("Success!")
except TypeError as e:
    print(f"Caught expected TypeError: {e}")
except Exception as e:
    print(f"Caught unexpected error: {type(e).__name__}: {e}")
