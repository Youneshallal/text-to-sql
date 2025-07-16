import json
import re
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, inspect
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLaMA3 model through Ollama
model = OllamaLLM(model="llama3")

# Define your database
db_url = "sqlite:///northwind_small.sqlite"

# SQL generation prompt template
sql_prompt_template = """
You are a SQL generator. Given a schema and a user question, you MUST output ONLY a valid SQL query if possible.No explanation is needed.
If the question is irrelevant to the schema or cannot be answered with it, return exactly: please ask again

Examples:

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


# Ask LLaMA3 to return a JSON list of relevant table names
def get_relevant_tables(question: str, table_details: str) -> list[str]:
    prompt = f"""
You are a precise and helpful assistant specialized in selecting ONLY the relevant SQL tables needed to answer a user's question.

Below are some examples:

User Question:
List all customers along with the number of orders they placed.

Relevant Tables:
["Customer", "Order"]

User Question:
Show the names and countries of all customers who placed more than 5 orders in 1997, where at least one of their orders was shipped by 'Speedy Express'.

Relevant Tables:
["Customer", "Order", "Shipper"]

User Question:
Which employees work in territories that belong to the 'Western' region?

Relevant Tables:
["Employee", "EmployeeTerritory", "Territory", "Region"]

User Question:
List the top 3 customers by total purchase amount.

Relevant Tables:
["Customer", "Order", "OrderDetail"]

User Question:
List the names of products that are currently out of stock.

Relevant Tables:
["Product"]

User Question:
Show each product's name, its category, and the supplier company.

Relevant Tables:
["Product", "Category", "Supplier"]

User Question:
Which employees have not handled any orders?

Relevant Tables:
["Employee", "Order"]

User Question:
List the names of shippers and how many orders they have shipped.

Relevant Tables:
["Order", "Shipper"]

User Question:
Find the names of customers who ordered products from the 'Seafood' category.

Relevant Tables:
["Customer", "Order", "OrderDetail", "Product", "Category"]

User Question:
Get the average unit price of all products supplied by each supplier.

Relevant Tables:
["Product", "Supplier"]

Now, your task:

User Question:
{question}

Below are the available tables with their descriptions:
{table_details}

Return ONLY a JSON array (list) of table names relevant to answering the question.
Use exact table names as given.
Do NOT include any explanation, comments, or extra text.
If no table is relevant, return an empty JSON array [].

Respond with ONLY the JSON array.
"""
    response = model.invoke(prompt)

    try:
        tables = json.loads(response.strip())
        if isinstance(tables, list) and all(isinstance(t, str) for t in tables):
            return tables
    except json.JSONDecodeError:
        pass

    return []



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

# Text input
user_query = st.chat_input("Describe what data you want...")

if user_query:
    st.session_state.history.append(("user", user_query))
    table_details = get_table_details()
    relevant_tables = get_relevant_tables(user_query, table_details)

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
