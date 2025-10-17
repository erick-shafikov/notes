# CREATE

Создание таблицы

```sql
CREATE TABLE products ( //создаем таблицу
  id INT NOT NULL, //id, который ниже будет определен, как уникальный идентификатор не может быть нулем
  name STRING, //тип данных
  price MONEY, //тип данных
  PRIMARY KEY(id) //задаем уникальный ID
)
```

# SELECT

Чтение данных из таблицы

```sql
SELECT * FROM products //выбрать всю таблицу, * обозначает выборку всех столбцов
SELECT name, price FROM products //выбрать только несколько столбцов
SELECT * FROM products WHERE id=1 //выбрать ряд с id = 1 из всей таблицы
SELECT count(*) FROM products //считаем сколько строк
SELECT count(*) as cnt FROM products //считаем сколько строк и добавляет столбец cnt
SELECT * FROM `products` WHERE id_cat=1 AND sale>0 //составные выборки
SELECT * FROM `products` WHERE id_cat=1 GROUP BY id_cat //группировка
SELECT * FROM `products` WHERE id_cat IN (SELECT DISTINCT id_cat FROM products WHERE sale > 0)//вложенный запрос

-- c сортировкой
SELECT * FROM `products` ORDER BY dt_add //выдаем сортировку по полю
SELECT * FROM `products` ORDER BY dt_add DESC, id_products DESC //выдаем сортировку по двум полям

```

обновление состояния
Вставляем данные

```sql
-- Вставляем данные
INSERT INTO products
VALUES (2, "Pencil", 0.80, 12)
-- выбрать для обновления значения поля price в ячейке с id=2
UPDATE products
SET price = 0.80
WHERE id=2
-- добавили столб INT переменной типа число
ALTER TABLE products
ADD stock INT
-- удалить
DELETE FROM products
WHERE id=2
```

# JOIN

```sql

SELECT * FROM `products` join cats ON products.id_cat = cats.id_cat или SELECT * FROM `products` join cats USING (id_cat)
-- //выбрать перекрёстную таблицу без произведения таблиц

```

# SQL. FOREIGN

```sql
CREATE TABLE orders (
  id INT NOT NUll,
  order_number INT,
  customer_id INT,
  product_id INT,
  PRIMARY KEY (id), //создаем таблицу
  FOREIGN KEY (customer_id) REFERENCES customers(id), //завязываем поле customer_id в таблице orders с полем id в таблице customers
  FOREIGN KEY (product_id) REFERENCES products(id)
)

JOIN
SELECT orders.order_number, customers.first_name, customers.last_name, customers.adress
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id

```
