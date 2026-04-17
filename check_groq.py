import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_connection():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("ERROR: GROQ_API_KEY is not set correctly in the .env file.")
        return

    client = Groq(api_key=api_key)
    
    print("Connecting to Groq...")
    try:
        # Simple test call
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10
        )
        print("SUCCESS: Connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"FAILED: Connection failed: {e}")

if __name__ == "__main__":
    check_connection()
