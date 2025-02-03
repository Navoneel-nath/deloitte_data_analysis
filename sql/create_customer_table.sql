DROP TABLE IF EXISTS customer_orders;

CREATE TABLE customer_orders (
    order_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    customer_number VARCHAR(50),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    city VARCHAR(50),
    order_date DATE,
    order_time TIME,
    region VARCHAR(50),
    sales DECIMAL(10,2),
    discount DECIMAL(5,2),
    profit DECIMAL(10,2),
    state VARCHAR(50),
    age INT
);
