-- Drop existing tables if they exist (for re-run)
DROP TABLE IF EXISTS clients;
DROP TABLE IF EXISTS orders;

-- Create clients table
CREATE TABLE clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
);

-- Create commands table with order_date column
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER,
    product TEXT NOT NULL,
    price REAL NOT NULL,
    order_date TEXT NOT NULL,  -- Storing date as TEXT in ISO format
    FOREIGN KEY (client_id) REFERENCES clients(id)
);

-- Insert example data
INSERT INTO clients (name, email) VALUES
    ('Alice Dupont', 'alice@example.com'),
    ('Bob Martin', 'bob@example.com');

INSERT INTO orders (client_id, product, price, order_date) VALUES
    (1, 'Laptop', 999.99, '2024-06-10'),
    (2, 'Smartphone', 599.49, '2024-06-12'),
    (1, 'Mouse', 25.00, '2024-06-14');
