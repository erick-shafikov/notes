# subqueries

# subqueries в WHERE

запросы, которые используют результаты другого запроса

```sql
-- пример показать все записи значения amount которых больше среднего значения amount по всей таблице
SELECT *
FROM payment
WHERE amount > (SELECT AVG(amount) FROM payment)
```

```sql
-- выбрать из payment всех кто в таблице customer first_name = ADAM
SELECT *
FROM payment
WHERE customer_id = (
	SELECT customer_id FROM customer
	WHERE first_name = 'ADAM'
	)

-- выбрать из payment всех кто в таблице customer first_name начинается с a
SELECT
*
FROM payment
WHERE customer_id IN (
	SELECT customer_id FROM customer
	WHERE first_name LIKE 'A%'
	)
```

```sql
-- выбрать все film у которых количество в store_id = 2 больше 3
SELECT * FROM film WHERE film_id IN
(SELECT film_id FROM inventory
WHERE store_id = 2
GROUP BY film_id
HAVING COUNT(*) > 3)
```

```sql
SELECT
ROUND(AVG(amount_per_day))
FROM
(SELECT
SUM(amount) as amount_per_day,
DATE(payment_date)
FROM payment
GROUP BY DATE(payment_date)) A
```

# subqueries в FROM

для двойных вычислений

```sql
-- (2) считаем среднее по суммам
SELECT ROUND(AVG(total_amount), 2) as avg_lifetime_spent
FROM
-- (1) считаем сумму по customer_id
(SELECT customer_id, SUM(amount) as total_amount FROM payment
GROUP BY customer_id) AS subquery -- если используем саб-запросы в FROM, то нужно именовать [subquery]
```

# subqueries в SELECT

добавит столбец в конец таблицы

```sql
-- посчитать разницу между amount строки и максимальным amount
SELECT
* , (SELECT MAX(amount) FROM payment) - amount as diff
FROM payment
```

# correlated subqueries в WHERE

нужен запрос который будет ссылать на общий показатель по группам в таблице

вычисляется для каждого ряда отдельно

```sql
-- e1 = employees
SELECT first_name, sales FROM employees e1
-- где столбец sales
WHERE sales > (
  -- больше среднего sales
	SELECT AVG(sales) FROM employees e2
  -- сравнимые если одинаковые по столбцу city
	WHERE e1.city = e2.city
)

```

# correlated subqueries в SELECT

добавит минимальное значение по строкам, где выполнялось бы условие

```sql

SELECT first_name, sales, (
	SELECT MIN(sales) FROM employee e3
	WHERE e1.city = e3.city
)
FROM employees e1
WHERE sales (
	SELECT AVG(sales) FROM employees e2
	WHERE e1.city = e2.city
)
```

```sql
-- выбрать максимальный replacement_cost из рейтинга
SELECT title, rating,
-- среднее значение replacement_cost
(SELECT AVG(replacement_cost) FROM film f2
WHERE f1.rating = f2.rating)
FROM film f1
-- из наибольшего replacement_cost по rating
WHERE replacement_cost = (
SELECT MAX(replacement_cost) FROM film f3
WHERE f1.rating =f3.rating
)
```

```sql
-- c INNER JOIN
SELECT first_name, amount, payment_id
FROM payment p1
INNER JOIN customer c
ON p1.customer_id= c.customer_id
WHERE amount = (
SELECT MAX(amount) from payment p2
WHERE p1.customer_id = p2.customer_id
)
```
