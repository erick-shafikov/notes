# CASE

```sql
-- THEN что будет в колонке
CASE
WHEN condition THEN result
WHEN condition THEN result
WHEN condition THEN result
ELSE
END
```

```sql
SELECT
amount,
CASE
WHEN amount < 2 THEN 'low amount'
WHEN amount < 5 THEN 'medium amount'
ELSE 'hight amount'
END
FROM payment

-- пример 2
SELECT
TO_CHAR(book_date, 'Dy'),
TO_CHAR(book_date, 'Mon'),
CASE
WHEN TO_CHAR(book_date, 'Dy') = 'Mon' THEN 'Monday'
WHEN TO_CHAR(book_date, 'Mon') = 'Jul' THEN 'July'
END
FROM bookings
```

```sql
SELECT
COUNT(*) AS flights,
CASE
WHEN actual_departure is null THEN 'no departure time'
WHEN actual_departure-scheduled_departure < '00:05' THEN 'On time'
WHEN actual_departure-scheduled_departure < '01:00' THEN 'litle bit late'
ELSE 'late'
END as is_late -- это группировка по case END as __group_name__
FROM flights
GROUP BY is_late
```

```sql
SELECT
--...
CASE
WHEN actual_departure is null THEN 'no departure time'
--...
ELSE 'late'
END as is not NULL -- убрать всех кто не попал
FROM flights

```

посчитать количество определенных групп

```sql
SELECT
SUM (CASE WHEN rating IN ('PG', 'G') THEN 1 ELSE 0 END) AS no_ratings
FROM film
```

распределить по столбцам количество элементов

```sql
SELECT
SUM (CASE WHEN rating = 'PG' THEN 1 ELSE 0 END) as "PG",
SUM (CASE WHEN rating = 'NC-17' THEN 1 ELSE 0 END) as "NC-17",
SUM (CASE WHEN rating = 'R' THEN 1 ELSE 0 END) as "R",
SUM (CASE WHEN rating = 'PG-13' THEN 1 ELSE 0 END) as "PG-13",
SUM (CASE WHEN rating = 'PG' THEN 1 ELSE 0 END) as "PG"
FROM film
```
