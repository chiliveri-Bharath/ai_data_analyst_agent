import pandas as pd

def load_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)

    else:
        raise ValueError("Unsupported file type")

    return df