import os
from dotenv import load_dotenv
import json
import re
import pandas as pd
import streamlit as st

import numpy as np

from sqlalchemy import create_engine, inspect


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

st.set_page_config(page_title="NL to SQL", layout="wide")

load_dotenv()
gemini_api_key= os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(model_name='gemini-2.5-pro')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


# Define your database
db_url = "sqlite:///northwind_small.sqlite"

# SQL generation prompt template
sql_prompt_template = """
You are a senior data engineer helping users write correct and efficient SQL queries.

Your task is to write a syntactically correct SQL query using the provided database schema. You must understand the user's intent and take into account the conversation history.

Please follow these rules:
1. Use only the table and column names that are explicitly listed in the schema below.
2. Do not make up any tables or columns.
3. Do not explain the query or add any commentary — just return the raw SQL query.
4. If aggregation is needed, make sure to use appropriate `GROUP BY` clauses.
5. Format the SQL query nicely across multiple lines.

Conversation History:
{conversation_history}

Current User Request:
{query}

Database Schema (JSON format):
{schema}

SQL Query:
"""




# Function to extract schema based on relevant tables
def extract_schema(db_url, only_tables=None):
    engine = create_engine(db_url)
    inspector = inspect(engine)
    schema = {}

    for table in inspector.get_table_names():
        if only_tables and table not in only_tables:
            continue
        columns = inspector.get_columns(table)
        schema[table] = [col['name'] for col in columns]

    return json.dumps(schema, indent=2)


def clean_text(text: str):
    # Remove <think>...</think> tags
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown code fences ```sql ... ```
    cleaned_text = re.sub(r"```sql|```", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()



def build_conversational_context(history, max_turns=10):

    #Construit un historique conversationnel sous forme de dialogue pour le prompt.

    turns = history[-max_turns:]
    dialogue = ""
    for sender, message in turns:
        if sender == "user":
            dialogue += f"User: {message}\n"
        else:
            # On simplifie la réponse pour éviter les blocs markdown et longs SQL dans le contexte
            first_line = message.split('\n')[0]
            dialogue += f"Assistant: {first_line.strip()}\n"
    return dialogue.strip()


# Function to build the SQL query from prompt
def to_sql_query(query, schema, history):
    context = build_conversational_context(history)
    prompt = sql_prompt_template.replace("{query}", query).replace("{schema}", schema).replace("{conversation_history}",
                                                                                               context)

    try:
        response = model.generate_content(prompt)
        print(response.text)
        return response.text
    except Exception as e:
        return f"-- Error generating SQL: {e}"


# Load table descriptions from CSV
def get_table_details():
    df = pd.read_csv("database_table_descriptions.csv")
    details = ""
    for _, row in df.iterrows():
        details += f"Table Name: {row['Table']}\nDescription: {row['Description']}\n\n"
    return details

@st.cache_data
def get_table_embeddings():
    df = pd.read_csv("database_table_descriptions.csv")
    tables = df["Table"].tolist()
    descriptions = df["Description"].tolist()
    vectors = embedding_model.encode(descriptions)
    return list(zip(tables, descriptions, vectors))



def get_relevant_tables_semantic(query: str, table_info: list, top_k: int = 7, threshold: float = 0.45) -> list[str]:
    query_vector = embedding_model.encode([query])[0]

    scores = []
    for table_name, description, vector in table_info:
        similarity = cosine_similarity([query_vector], [vector])[0][0]
        if similarity >= threshold:  #  Only include if above threshold
            scores.append((table_name, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]




# --- Streamlit UI ---

st.markdown(
    """
    <style>
    .stChatInput input {
        font-size: 16px;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .suggestions button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        font-size: 14px;
        border-radius: 20px;
        background-color: #efefef;
        border: none;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Natural Language to SQL Generator")

# Suggested prompts
suggested_questions = [
    "List all customers and their order counts.",
    "Show products that are out of stock.",
    "List top 3 customers by total purchase.",
    "Which employees work in the Western region?",
    "Which suppliers provide 'Seafood' products?",
    "How many orders did each shipper deliver?"
]

if "history" not in st.session_state:
    st.session_state.history = []

# Suggestions UI
with st.expander("💡 Suggested Prompts", expanded=True):
    cols = st.columns(len(suggested_questions))
    for i, question in enumerate(suggested_questions):
        if cols[i].button(question):
            st.session_state.query_input = question

if st.button("🗑️ Clear Chat History"):
    st.session_state.history = []

# Text input
user_query = st.chat_input("Describe what data you want...")

# If user clicked a suggestion, use it as the input
if not user_query and "query_input" in st.session_state:
    user_query = st.session_state.query_input
    del st.session_state.query_input  # Clean it so it doesn't re-trigger every run


if user_query:
    st.session_state.history.append(("user", user_query))
    table_info = get_table_embeddings()
    relevant_tables = get_relevant_tables_semantic(user_query, table_info, threshold=0.2)


    if not relevant_tables:
        st.session_state.history.append(("assistant", "⚠️ Could not detect relevant tables. Try rephrasing your question."))
    else:
        schema = extract_schema(db_url, only_tables=relevant_tables)
        sql = to_sql_query(user_query, schema, st.session_state.history)
        # Store the last SQL query in session state for later execution
        st.session_state["last_sql"] = sql
        output = f"**🧩 Relevant Tables Detected:**\n✅ {', '.join(relevant_tables)}\n\n**🧾 Generated SQL Query:**\n{sql}"
        st.session_state.history.append(("assistant", output))

# Show chat history
for sender, msg in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(msg)

# --- Show query results ---
if "last_sql" in st.session_state:
    with st.expander("📊 Run SQL and Show Results"):
        if st.button("▶️ Execute SQL Query"):
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    result_df = pd.read_sql(clean_text(st.session_state["last_sql"]), conn)
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"❌ Error running query: {e}")
