# JOINS

виды:

- inner join
- outer join
- left join
- right join

всегда нужна join-column, на которую можно ссылаться

# inner join (A & B)

для комбинирования по одной колонке, применятся для элементов, которые имеют ссылки в обеих таблицах, дублирующие колонки будут повторяться. Нет разница в каком порядке указывать. Синтаксис:

```sql
SELECT * FROM products
JOIN cats
ON products.id_cat = cats.id_cat

SELECT
*  FROM tableA
INNER JOIN tableB
ON tableA.employee = tableB.employee

-- с алиасами
SELECT
*  FROM tableA AS A
INNER JOIN tableB AS B
ON tabAleA.employee = B.employee

-- еще вариант если колонки называются одинаково
SELECT employee FROM tableA A
INNER JOIN tableB B
ON A.employee = B.employee

-- c указанием названия столбца
SELECT A.employee FROM tableA A
INNER JOIN tableB B
ON A.employee = B.employee
```

```sql
SELECT
pa.*, -- выбрать все из payment таблицы
first_name, -- из customer подтянуться только first_name, last_name
last_name
FROM payment pa --alias
INNER JOIN customer cu --alias
-- вариант без alias
-- ON payment.customer_id = customer.customer_id
ON pa.customer_id = cu.customer_id
```

# outer join (A || B)

позволяет получить все колонки, которые есть в двух таблицах. При объединении столбцы B, которые не имеют столбцов A будет обозначены как null

```sql
SELECT
*
FROM TableA
FULL OUTER JOIN TableB
ON TableA.employee = TableB.employee
```

```sql
-- A - (A & B)
SELECT * FROM tableA a
FULL OUTER JOIN tableB b
ON a.same_field = b.same_field
WHERE b.same_field IS null
```

# left outer join (A + A & B)

брать первую таблицу и только общее из второй

```sql
SELECT
*
FROM TableA
LEFT OUTER JOIN TableB
ON TableA.employee = TableB.employee
```

# right outer join (A + A & B)

брать вторую таблицу и только общее из первой. Это тот же left outer только таблицы поменяли в порядке

```sql
SELECT
*
FROM TableA
RIGHT OUTER JOIN TableB
ON TableA.employee = TableB.employee
```

# join нескольких колонок

```sql
SElECT * FROM TableA a
INNER JOIN Table b
ON a.column_1 = b.column_1
AND a.column_2 = b.column_2 -- более эффективно чем
```

```sql
SElECT * FROM TableA a
INNER JOIN Table b
ON a.column_1 = b.column_1
AND a.column_2 = 'some_value'
-- более эффективно чем WHERE AND a.column_2 = 'some_value'
```

[PK] - primary keys

# join нескольких таблиц

Порядок неважен дял INNER JOIN, но важен для LEFT и RIGHT JOIN

```sql
SELECT column_1, c.column_2 FROM table_a a
INNER JOIN table_b b
ON a.column_1 = b.column_1
INNER JOIN table_c c
ON b.column_2 = c.column_2
```

объединение трех таблиц

```sql
SELECT
    s.fare_conditions AS "Fare Conditions",
    COUNT(*) AS "Count"
FROM
    boarding_passes bp --алиас для таблицы 1
INNER JOIN
    flights f ON bp.flight_id = f.flight_id --join + алиас для таблиц 1 + 2
INNER JOIN
    seats s ON f.aircraft_code = s.aircraft_code AND bp.seat_no = s.seat_no --join + алиас для таблиц 2 + 3
GROUP BY -- сортировка
    s.fare_conditions
ORDER BY 2 DESC;
```

# self join

когда таблица ссылается сама на себя

```sql
-- Найти в таблице film строки с одинаковым length
SELECT f1.title, f2.title, f2.length FROM film f1
JOIN film f2
ON f1.length = f2.length
WHERE f1.film_id != f2.film_id
ORDER BY 3 DESC
```

# cross join

все возможные комбинации

```sql
SELECT
t1.column,
t1.column
FROM table1 t1
CROSS JOIN table2 t2
```

# natural join

автоматические соединяет колонки с одинаковым именем, позволит не использовать ON. Но если есть два одинаковых столбца без совпадений приведет к тому, что таблица будет пустая

```sql
SELECT
*
FROM table1
NATURAL LEFT JOIN table2
```

# USING

```sql
-- или
SELECT * FROM `products` join cats USING (id_cat)
```
