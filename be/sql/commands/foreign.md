# SQL. FOREIGN

```sql
CREATE TABLE orders (
  id INT NOT NUll,
  order_number INT,
  customer_id INT,
  product_id INT,
  PRIMARY KEY (id), --создаем таблицу
  FOREIGN KEY (customer_id) REFERENCES customers(id), --завязываем поле customer_id в таблице orders с полем id в таблице customers
  FOREIGN KEY (product_id) REFERENCES products(id)
)

JOIN
SELECT orders.order_number, customers.first_name, customers.last_name, customers.adress
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id

```
