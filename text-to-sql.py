import json
import re
import pandas as pd
import streamlit as st

import numpy as np

from sqlalchemy import create_engine, inspect
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize the LLaMA3 model through Ollama
model = OllamaLLM(model="llama3:latest")
# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define your database
db_url = "sqlite:///northwind_small.sqlite"

# SQL generation prompt template
sql_prompt_template = """
You are a SQL generator. Given a database schema and a user question, your task is to output ONLY a valid **SQLite** SQL query that correctly answers the question using the schema. 

❗Rules you MUST follow:
- Output ONLY SQL code, and nothing else.
- Use ONLY the tables and columns that are **explicitly defined** in the provided schema.
- If a table or column required to answer the question is missing from the schema, return EXACTLY: please ask again
- If the question is irrelevant to the schema or impossible to answer with the available data, return EXACTLY: please ask again
- Never assume columns or tables that are not listed. Be strict.

🧠 Tip: Always read the schema carefully before generating the query.

Examples:

Schema:
{{ "Employee": ["Id", "FirstName", "LastName"], "EmployeeTerritory": ["EmployeeId", "TerritoryId"], "Territory": ["Id", "RegionId"], "Region": ["Id", "RegionDescription"] }}
User question:
Which employees work in territories that belong to the 'Western' region?
Output:
SELECT e.FirstName, e.LastName
FROM Employee e
JOIN EmployeeTerritory et ON e.Id = et.EmployeeId
JOIN Territory t ON et.TerritoryId = t.Id
JOIN Region r ON t.RegionId = r.Id
WHERE r.RegionDescription = 'Western';

Schema:
{{ "Customer": ["Id", "CompanyName", "Country"], "Order": ["Id", "CustomerId", "OrderDate", "ShipVia"], "Shipper": ["Id", "CompanyName"] }}
User question:
Show customers who placed more than 10 orders in 1998 that were shipped by 'Federal Shipping'.
Output:
SELECT c.CompanyName, COUNT(o.Id) AS OrderCount
FROM Customer c
JOIN Order o ON c.Id = o.CustomerId
JOIN Shipper s ON o.ShipVia = s.Id
WHERE o.OrderDate >= '1998-01-01' AND o.OrderDate < '1999-01-01'
AND s.CompanyName = 'Federal Shipping'
GROUP BY c.CompanyName
HAVING COUNT(o.Id) > 10;

Schema:
{{ "Supplier": ["Id", "CompanyName"], "Product": ["Id", "SupplierId", "ProductName", "UnitsInStock", "CategoryId"], "Category": ["Id", "CategoryName"] }}
User question:
List suppliers who provide products in the 'Beverages' category that are currently out of stock.
Output:
SELECT s.CompanyName
FROM Supplier s
JOIN Product p ON s.Id = p.SupplierId
JOIN Category c ON p.CategoryId = c.Id
WHERE c.CategoryName = 'Beverages' AND p.UnitsInStock = 0;

Schema:
{{ "Order": ["Id", "CustomerId", "OrderDate"], "OrderDetail": ["OrderId", "ProductId", "UnitPrice", "Quantity", "Discount"], "Product": ["Id", "ProductName", "CategoryId"], "Category": ["Id", "CategoryName"], "Customer": ["Id", "CompanyName", "Country"] }}
User question:
Find customers from Germany who spent more than $10,000 in total on 'Seafood' products in 1997.
Output:
SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalSpent
FROM Customer c
JOIN Order o ON c.Id = o.CustomerId
JOIN OrderDetail od ON o.Id = od.OrderId
JOIN Product p ON od.ProductId = p.Id
JOIN Category cat ON p.CategoryId = cat.Id
WHERE c.Country = 'Germany'
AND cat.CategoryName = 'Seafood'
AND o.OrderDate >= '1997-01-01' AND o.OrderDate < '1998-01-01'
GROUP BY c.CompanyName
HAVING TotalSpent > 10000;

Schema:
{{ "Employee": ["Id", "FirstName", "LastName", "Title"], "Order": ["Id", "EmployeeId", "OrderDate"], "OrderDetail": ["OrderId", "ProductId", "Quantity"], "Product": ["Id", "ProductName", "CategoryId"], "Category": ["Id", "CategoryName"] }}
User question:
Which employees handled orders that included products from the 'Confections' category in the year 1996?
Output:
SELECT DISTINCT e.FirstName, e.LastName
FROM Employee e
JOIN Order o ON e.Id = o.EmployeeId
JOIN OrderDetail od ON o.Id = od.OrderId
JOIN Product p ON od.ProductId = p.Id
JOIN Category c ON p.CategoryId = c.Id
WHERE c.CategoryName = 'Confections'
AND o.OrderDate >= '1996-01-01' AND o.OrderDate < '1997-01-01';

Schema:
{{ "Customer": ["Id", "CompanyName"], "Order": ["Id", "CustomerId", "OrderDate"], "OrderDetail": ["OrderId", "ProductId", "Quantity"], "Product": ["Id", "ProductName", "SupplierId"], "Supplier": ["Id", "CompanyName"] }}
User question:
List customers who ordered products from at least 3 different suppliers in 1999.
Output:
SELECT c.CompanyName
FROM Customer c
JOIN Order o ON c.Id = o.CustomerId
JOIN OrderDetail od ON o.Id = od.OrderId
JOIN Product p ON od.ProductId = p.Id
WHERE o.OrderDate >= '1999-01-01' AND o.OrderDate < '2000-01-01'
GROUP BY c.CompanyName
HAVING COUNT(DISTINCT p.SupplierId) >= 3;

Schema:
{{ "Customer": ["Id", "CompanyName", "Country"], "Order": ["Id", "CustomerId"] }}
User question:
List all customers along with the number of orders they placed.
Output:
SELECT c.CompanyName, COUNT(o.Id) AS OrderCount
FROM Customer c
LEFT JOIN Order o ON c.Id = o.CustomerId
GROUP BY c.CompanyName;

Schema:
{{ "Product": ["Id", "ProductName", "UnitsInStock"] }}
User question:
List the names of products that are out of stock.
Output:
SELECT ProductName
FROM Product
WHERE UnitsInStock = 0;

Schema:
{{ "Customer": ["Id", "CompanyName", "Country"], "Order": ["Id", "CustomerId", "OrderDate", "ShipVia"], "Shipper": ["Id", "CompanyName"] }}
User question:
Show the names and countries of all customers who placed more than 5 orders in 1997, where at least one of their orders was shipped by 'Speedy Express'.
Output:
SELECT c.CompanyName, c.Country
FROM Customer c
JOIN Order o ON c.Id = o.CustomerId
JOIN Shipper s ON o.ShipVia = s.Id
WHERE o.OrderDate >= '1997-01-01' AND o.OrderDate < '1998-01-01'
  AND s.CompanyName = 'Speedy Express'
GROUP BY c.CompanyName, c.Country
HAVING COUNT(o.Id) > 5;

Schema:
{{ "Employee": ["Id", "FirstName", "LastName"], "EmployeeTerritory": ["EmployeeId", "TerritoryId"], "Territory": ["Id", "RegionId"], "Region": ["Id", "RegionDescription"] }}
User question:
Which employees work in territories that belong to the 'Western' region?
Output:
SELECT e.FirstName, e.LastName
FROM Employee e
JOIN EmployeeTerritory et ON e.Id = et.EmployeeId
JOIN Territory t ON et.TerritoryId = t.Id
JOIN Region r ON t.RegionId = r.Id
WHERE r.RegionDescription = 'Western';

Schema:
{{ "Customer": ["Id", "CompanyName"], "Order": ["Id", "CustomerId"], "OrderDetail": ["OrderId", "UnitPrice", "Quantity", "Discount"] }}
User question:
List the top 3 customers by total purchase amount.
Output:
SELECT c.CompanyName, 
       SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalAmount
FROM Customer c
JOIN Order o ON c.Id = o.CustomerId
JOIN OrderDetail od ON o.Id = od.OrderId
GROUP BY c.CompanyName
ORDER BY TotalAmount DESC
LIMIT 3;

-------------------------

Schema:
{schema}

User question:
{query}

Output (SQL only, or 'please ask again'):
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


# Function to clean the LLM response
def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


# Function to build the SQL query from prompt
def to_sql_query(query, schema):
    prompt = ChatPromptTemplate.from_template(sql_prompt_template)
    chain = prompt | model
    return clean_text(chain.invoke({"query": query, "schema": schema}))


# Load table descriptions from CSV
def get_table_details():
    df = pd.read_csv("database_table_descriptions.csv")
    details = ""
    for _, row in df.iterrows():
        details += f"Table Name: {row['Table']}\nDescription: {row['Description']}\n\n"
    return details

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
st.set_page_config(page_title="NL to SQL", layout="wide")
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
        sql = to_sql_query(user_query, schema)
        output = f"**🧩 Relevant Tables Detected:**\n✅ {', '.join(relevant_tables)}\n\n**🧾 Generated SQL Query:**\n```sql\n{sql}\n```"
        st.session_state.history.append(("assistant", output))

# Show chat history
for sender, msg in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(msg)
