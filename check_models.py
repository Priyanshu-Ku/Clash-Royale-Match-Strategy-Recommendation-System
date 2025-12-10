import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the .env file to get your key
try:
    load_dotenv(".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        exit()
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error loading .env or configuring API: {e}")
    exit()

print("--- Available Gemini Models ---")
print("This will list all models your API key can access:\n")

try:
    for model in genai.list_models():
        # We only care about models that support 'generateContent'
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model name: {model.name}")
            print(f"  Description: {model.description}")
            print("-" * 20)

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An error occurred while listing models: {e}")
    print("This could be due to an invalid API key or a network issue.")

print("\n--- End of List ---")
print("\nPlease copy/paste this list. We need to find the correct model name (e.g., 'models/gemini-pro').")