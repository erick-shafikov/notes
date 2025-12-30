# window функции

нужны для обхода саб-запросов и группировок

```sql
--                     окно агрегации
AGG (agg_column) OVER(PARTITION BY partition_column)
```

AGG:

- SUM
- COUNT

# OVER + PARTITION BY

```sql
SELECT
transaction_id,
payment_type,
customer_id
price_in_transaction,
(
  SELECT SUM(price_in_transaction)
  FROM sales s2
  WHERE s2.customer_id=s1.customer_id
)
FROM sales s1
```

```sql
SELECT
transaction_id,
payment_type,
customer_id
price_in_transaction,
-- добавить новый столбец суммы price_in_transaction по customer_id
SUM(price_in_transaction) OVER (PARTITION BY customer_id)
-- или
COUNT(*) OVER (PARTITION BY payment_type)
FROM sales s
```

# OVER + ORDER BY

вычисление скользящим итогом. ТО есть будут сложены все нарастающим итого

```sql
-- сумма будет вычисляться по датам
SELECT  *,
SUM(amount) OVER(ORDER BY payment_date)
FROM payment
```

комбинация PARTITION BY + ORDER BY

```sql
SELECT  *,
-- сумма будет вычисляться по payment_id и по customer_id
SUM(amount) OVER(PARTITION BY customer_id ORDER BY payment_id)
FROM payment
```

# RANK DENSE_RANK

для ранжирования

```sql
SELECT
f.title,
c.name,
f.length,
-- по названия категории с ранжирует по length,
-- без PARTITION BY name ранжирует всех по length
DENSE_RANK() OVER(PARTITION BY name ORDER BY length DESC)
FROM film f
LEFT JOIN film_category fc ON f.film_id = fc.film_id
LEFT JOIN category c ON c.category_id = fc.category_id
```

```sql
-- выбрать из каждой страны country первые 3, customer_id
SELECT * FROM
(SELECT
name,
country,
COUNT(*),
RANK() OVER(PARTITION BY country ORDER BY COUNT(*) DESC) AS rank --aлиас для условия ниже
FROM customer_list
LEFT JOIN payment
ON id=customer_id
GROUP BY name, country) a
WHERE rank BETWEEN 1 AND 3
```

# VALUE

```sql
SELECT * FROM
(SELECT
name,
country,
COUNT(*),
-- отобразит имя того кто имеет первое место в группе по country
FIRST_VALUE(name) OVER(PARTITION BY country ORDER BY COUNT(*) ASC) AS rank
FROM customer_list
LEFT JOIN payment
ON id=customer_id
GROUP BY name, country) a
```

# LEAD LAG

указатели на следующие записи

```sql
SELECT * FROM
(SELECT
name,
country,
COUNT(*),
-- отобразит количество предыдущего в группе
LEAD(COUNT(*)) OVER(PARTITION BY country ORDER BY COUNT(*) ASC) AS rank,
-- отобразит разницу количество предыдущего и себя в группе
LEAD(COUNT(*)) OVER(PARTITION BY country ORDER BY COUNT(*) ASC) - COUNT(*)

FROM customer_list
LEFT JOIN payment
ON id=customer_id
GROUP BY name, country) a
```
