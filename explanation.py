import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "your_groq_api_key_here":
    client = None
else:
    client = Groq(api_key=api_key)

def generate_explanation(task, df, col1=None, col2=None):
    """
    Generates a dynamic explanation using Groq's LLM based on the task and data context.
    """
    if not client:
        return [f"Analysis completed for {task} on {col1}."]
    
    # Construct a concise context for the AI
    data_context = {
        "task": task,
        "column1": col1,
        "column2": col2,
        "shape": df.shape,
        "sample_data": df[[col for col in [col1, col2] if col in df.columns]].head(3).to_dict() if col1 or col2 else "N/A"
    }

    if col1 and col1 in df.columns and df[col1].dtype != 'O':
        data_context["mean"] = df[col1].mean()
        data_context["max"] = df[col1].max()
        data_context["min"] = df[col1].min()

    prompt = f"""
You are a senior data analyst. Explain the following data task to a user:
- Task: {task}
- Context: {data_context}

Provide 2-3 concise, high-value bullet points explaining what the visualization or analysis shows and what the key takeaway is.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        explanation = response.choices[0].message.content
        # Split by newlines and clean
        lines = [line.strip().lstrip("👉 ").lstrip("- ").lstrip("* ") for line in explanation.split("\n") if line.strip()]
        return lines
    except Exception as e:
        return [f"Analysis completed for {task} on {col1}."]
