import pandas as pd

data = [
    (
        "Category",
        "Stores product category details, including unique category ID, category name, and a textual description that classifies products into logical groups."
    ),
    (
        "Customer",
        "Contains detailed customer information such as unique customer ID, company name, contact person’s name and title, full address (street, city, region, postal code, country), phone and fax numbers."
    ),
    (
        "CustomerCustomerDemo",
        "Associative linking table connecting customers to their demographic segments, mapping customer IDs to customer type IDs for classification."
    ),
    (
        "CustomerDemographic",
        "Defines customer demographic types or market segments, with a unique ID and a descriptive field outlining the characteristics of each customer group."
    ),
    (
        "Employee",
        "Stores company employee records, including employee ID, names (first and last), job title, courtesy title, birth and hire dates, contact information, address, manager (ReportsTo), and notes."
    ),
    (
        "EmployeeTerritory",
        "Maps employees to the sales territories they are responsible for, linking employee IDs with territory IDs to define coverage areas."
    ),
    (
        "Order",
        "Records customer orders with order ID, linked customer and employee IDs, order and shipping dates, shipper ID (ShipVia), freight cost, and detailed shipping address information."
    ),
    (
        "OrderDetail",
        "Details line items of each order, linking orders to products, including quantity ordered, unit price at order time, and any applicable discount."
    ),
    (
        "Product",
        "Stores individual product data including product ID, product name, supplier and category IDs, quantity per unit description, unit price, stock levels (units in stock and on order), reorder level, and discontinued status."
    ),
    (
        "Region",
        "Defines broader geographic sales regions with region ID and descriptive region name, used to group sales territories."
    ),
    (
        "Shipper",
        "Contains shipping company information including shipper ID, shipping company name, and contact phone number, used to identify companies responsible for delivering orders."
    ),
    (
        "Supplier",
        "Stores supplier details such as supplier ID, company name, contact person’s name and title, full address, phone, fax, and homepage URL."
    ),
    (
        "Territory",
        "Defines sales territories by territory ID, descriptive name, and links to a geographic region via region ID for sales management."
    ),
]


df = pd.DataFrame(data, columns=["Table", "Description"])
df.to_csv("database_table_descriptions.csv", index=False, encoding="utf-8")
print("CSV file generated successfully.")
