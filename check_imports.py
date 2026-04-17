import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import google.generativeai as genai
    print("SUCCESS: google.generativeai imported")
    print(f"google.generativeai version: {genai.__version__}")
except ImportError as e:
    print(f"ERROR: google.generativeai not found: {e}")

try:
    from dotenv import load_dotenv
    print("SUCCESS: dotenv imported")
except ImportError as e:
    print(f"ERROR: dotenv not found: {e}")

try:
    import pandas as pd
    print("SUCCESS: pandas imported")
except ImportError as e:
    print(f"ERROR: pandas not found: {e}")

try:
    import streamlit as st
    print("SUCCESS: streamlit imported")
except ImportError as e:
    print(f"ERROR: streamlit not found: {e}")
