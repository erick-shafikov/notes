# CREATE

При создании таблицы нужно указать название колонок, тип и ограничения.
Ограничения:

- NOT NULL - Запрещает NUL
- UNIQUE - Значение в столбце должно быть уникальным
- DEFAULT- Значение по умолчанию
- PRIMARY KEY = NOT NULL + UNIQUE - что отличает ряд из таблицы, уникальный идентификатор строки, не может быть NULL
- REFERENCES
- FOREIGN KEY - ссылка на PRIMARY KEY другой таблицы
- [CHECK- Условие, которое должно выполняться](#check)

```sql
CREATE DATABASE data_base_name; -- создаст БД, за основу будет взята template1
CREATE DATABASE "company y" -- с пробелом
WITH encoding = 'UTF-8' -- c настройкой кодировки
COMMENT ON DATABASE company_x IS 'database description' -- с описанием
CREATE DATABASE data_base_name TEMPLATE template_db; -- создаст БД, за основу будет взята template_db
```

```sql
CREATE TABLE products ( -- создаем таблицу
  id INT NOT NULL, -- id, который ниже будет определен, как уникальный идентификатор не может быть нулем
  name STRING, -- тип данных
  price MONEY, -- тип данных
  PRIMARY KEY(id) -- задаем уникальный ID
)

--еще пример
CREATE TABLE online_sales (
  transaction_id SERIAL PRIMARY KEY,
  customer_id INT REFERENCES customer(customer_id),
  film_id INT REFERENCES film(film_id),
  amount numeric(5,2) NOT NULL,
  promotion_code VARCHAR(10) DEFAULT 'None'
)

--еще пример
CREATE TABLE director (
  director_id SERIAL PRIMARY KEY,
  director_account_name VARCHAR(20) UNIQUE,
  first_name VARCHAR(50),
  last_name VARCHAR(50) DEFAULT 'Not specified',
  date_of_birth DATE,
  address INT REFERENCES address(address_id) --ссылка на таблицу address
)
```

пример с FOREIGN b jnj,hf;tybt

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
SELECT orders.order_number, customers.first_name, customers.last_name, customers.address
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id

```

# INSERT

```sql
-- если нужно вставить по очереди
INSERT INTO table_name
VALUES (value_1, value_2,...)

-- если нужно вставить не по очереди, нудно помнить про ограничения
INSERT INTO table_name
(column_1, column_2, column_4)
VALUES (value_1, value_2,value_4)

INSERT INTO table_name
(column_1, column_2, column_4) -- если нужно вставить несколько строк
VALUES (value_1, value_2,value_4),(value_1, value_2,value_4)
```

# ALTER TABLE

Можно добавлять, удалять, переименовывать колонки, изменять тип

```sql
 -- удаление
ALTER TABLE table_name
DROP COLUMN column_name
-- удаление c проверкой
ALTER TABLE table_name
DROP COLUMN IF EXISTS column_name
```

```sql
-- добавить
ALTER TABLE table_name
ADD COLUMN new_column DATE

-- добавление c проверкой
ALTER TABLE table_name
ADD COLUMN IF NOT EXISTS new_column DATE
```

```sql
-- изменить тип
ALTER TABLE table_name
ALTER COLUMN column_name TYPE NEW_TYPE -- SMALLINT дял примера
```

```sql
-- переименовать
ALTER TABLE table_name
RENAME COLUMN old_name TO new_name
-- или
ALTER TABLE old_table_name
RENAME TO new_table_name
```

```sql
-- значение по умолчанию
ALTER TABLE table_name
ALTER COLUMN column_name SET DEFAULT some_value
```

```sql
ALTER TABLE table_name
ALTER COLUMN column_name SET NOT NULL
```

```sql
-- сброс ограничения
ALTER TABLE table_name
ALTER COLUMN column_name DROP NOT NULL
-- добавление ограничения
ALTER TABLE table_name
ADD CONSTRAINT column_name UNIQUE(column1)
-- PK и несколько команд
ALTER TABLE table_name
ADD CONSTRAINT column_name
ADD PRIMARY KEY(column_1, column_2...)
```

# DROP

Удаляет таблицу, схему

```sql
DROP TABLE table_name
DROP SCHEMA schema_name
```

# TRUNCATE

Удаляет все содержимое

```sql
TRUNCATE table_name
TRUNCATE schema_name
```

# CHECK

для создания пользовательских ограничений

```sql
CREATE TABLE table_name (
  column_name TYPE CHECK(condition)
)
```

```sql

CREATE TABLE director (
  name TEXT CHECK (length(name) > 1)
)
-- дать ограничению имя <table>_<column>_check имя ограничения по умолчанию
CREATE TABLE director (
  name TEXT CONSTRAINT name_length CHECK (length(name) > 1)
)

-- добавление в текущую таблицу
ALTER TABLE director
ADD CONSTRAINT data_check CHECK(start_date < end_date)

-- переименование ограничения
ALTER TABLE director
RENAME CONSTRAINT data_check TO data_constraint
```

# CREATE AS

позволяет создать таблицу из запроса

```sql
CREATE TABLE table_name AS query
```

```sql
-- создаст полную копию customer и назовет customer_test
CREATE TABLE customer_test
AS
SELECT * FROM customer

-- c WHERE
CREATE TABLE customer_test
AS
SELECT customer_id, initials
FROM customer
WHERE first_name LIKE 'C%'
```

# VIEW

Позволяет привязать созданные таблицы из запроса, позволяя привязать данные созданное таблицы к данным таблиц из которых она была создана. Но таблицей она не является. Ее нет в памяти

```sql
CREATE VIEW customer_anonymous
AS
SELECT customer_id, initials
FROM customer
WHERE first_name LIKE 'C%'
-- обращение как к обычной таблице

SELECT * FROM customer_anonymous
```

view - медленные, так как запрос может быть медленный. Выход - создать таблицу из этой таблицы. Но тогда данные будут устаревшие

```sql
CREATE OR REPLACE VIEW v-customer
```

# MATERIALIZED VIEW

данные будут хранится в памяти, но данные нужно обновлять с помощью REFRESH

```sql
CREATE MATERIALIZED VIEW view_name
AS query

-- обновление данных
REFRESH MATERIALIZED VIEW
```

MATERIALIZED VIEW и VIEW поддерживают все операции ALTER, DROP

# IMPORT EXPORT

импорт и экспорт из таблиц csv файлов. Сначала таблица должна быть создана, потом загружен файл
