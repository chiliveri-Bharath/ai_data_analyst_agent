# 🤖 AI Data Analyst Agent

An AI-powered intelligent data analysis system that enables users to upload datasets and interact with them using natural language queries. The system automates data cleaning, exploration, visualization, and insight generation — acting like a virtual data analyst.

---

## 📌 Overview

The **AI Data Analyst Agent** is designed to simplify the data analysis process for both technical and non-technical users. Instead of writing complex queries or code, users can simply ask questions in plain English and receive meaningful insights, charts, and cleaned data.

This project demonstrates the integration of **Large Language Models (LLMs)** with traditional data analysis workflows.

---

## 🚀 Key Features

### 🧹 Data Cleaning & Preprocessing

* Automatic handling of missing values
* Removal of duplicates
* Data type correction
* Basic outlier handling

### 📊 Smart Data Visualization

* Automatic chart selection based on data types
* Supports:

  * Bar charts
  * Line charts
  * Histograms
  * Scatter plots
* Interactive visualizations using Plotly / Matplotlib

### 🧠 Natural Language Query System

* Ask questions like:

  * "What is the average sales?"
  * "Show top 5 products"
  * "Find correlation between variables"
* Converts user queries into actionable data operations

### ⚡ LLM Integration (Groq API)

* Fast response generation
* Context-aware data insights
* Enhances user interaction with data

### 📂 File Upload Support

* Supports CSV and Excel files
* Dynamic dataset loading

---

## 🛠️ Tech Stack

| Category             | Tools/Technologies |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Data Processing      | Pandas, NumPy      |
| Visualization        | Plotly, Matplotlib |
| Frontend UI          | Streamlit          |
| AI/LLM Integration   | Groq API           |

---

## 🏗️ System Architecture

1. User uploads dataset
2. Data preprocessing module cleans the data
3. User enters query in natural language
4. LLM interprets query
5. System executes analysis using Pandas
6. Results are displayed as:

   * Text insights
   * Tables
   * Charts

---

## 📂 Project Structure

ai-data-analyst-agent/
│── app.py                 # Main Streamlit application
│── data_cleaner.py        # Data preprocessing module
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── .gitignore             # Ignored files

---

## ⚙️ Installation & Setup

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ai-data-analyst-agent.git
cd ai-data-analyst-agent
```

### 🔹 Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Queries

* "Show summary statistics"
* "Plot sales over time"
* "Find missing values"
* "Top 10 highest revenue products"
* "Correlation between price and quantity"

---

## 📷 Demo

👉 Add:
<img width="1765" height="524" alt="image" src="https://github.com/user-attachments/assets/12e5654b-3085-496c-9564-b61f02a6fc5b" />
<img width="1799" height="709" alt="image" src="https://github.com/user-attachments/assets/a2ee550c-8ff3-4908-a344-6d5343bdb637" />
<img width="1710" height="444" alt="image" src="https://github.com/user-attachments/assets/a9483969-ce32-410e-9f60-b81b6804d938" />





* Screenshots of dashboard
* Sample charts
* Short demo video

---

## 📈 Future Enhancements

* 🔗 SQL database integration
* 📊 Power BI integration
* 🤖 Advanced ML model integration
* 🔐 User authentication system
* 🌐 Deployment (Streamlit Cloud / AWS)

---

## ⚠️ Limitations

* Performance depends on dataset size
* LLM responses may vary for complex queries
* Limited support for unstructured data

---

## 🤝 Contributing

Contributions are welcome!

Steps:

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push and create Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---


## 🔥 Project Highlights (For Recruiters)

* Real-world AI + Data Analytics integration
* End-to-end pipeline: Data → Cleaning → Analysis → Visualization
* Practical implementation of LLM-based analytics
* User-friendly interface using Streamlit


---

## 👨‍💻 Author

**Bharath Chiliveri**
MCA Graduate | Aspiring Data Scientist & AI Engineer

---


## ⭐ Support

If you find this project useful:

* Give it a ⭐ on GitHub
* Share it on LinkedIn

---


