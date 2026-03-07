# FUNCTION

```sql
CREATE OR REPLACE function_name (param1, param2)
  RETURN return_data_type
  LANGUAGE plpgsql [sql | c....]
AS
$$
DECLARE
<variable declaration>;
BEGIN
<function definition>;
END;
$$
```

Пример (x,y) => x + y + 3

```sql
CREATE first_function (c1 INT, c2 INT)
  RETURN INT
  LANGUAGE plpgsql
AS
$$
DECLARE
C3 INT;
BEGIN
SELECT c1 + c2 + 3
INTO c3;
RETURN c3;
END;
$$
```

пример с сслкой на таблицу

```sql
CREATE FUNCTION count_rental_rate(min_r decimal(4,2), max_r decimal(4,2))
RETURNS INT
LANGUAGE plpgsql
AS
$$
DECLARE
movie_count INT;
BEGIN
-- присвоение переменной
SELECT COUNT(*)
INTO movie_count
--
FROM film
WHERE rental_rate BETWEEN min_r AND max_r;
RETURN movie_count;
END;
$$
```

```sql
CREATE FUNCTION get_total_amount(fn TEXT, ln TEXT)
RETURNS INT
LANGUAGE plpgsql
AS
$$
DECLARE
  total_amount INT;
BEGIN
  SELECT COALESCE(SUM(p.amount), 0)
  INTO total_amount
  FROM payment p
  JOIN customer c
    ON c.customer_id = p.customer_id
  WHERE c.first_name = fn
    AND c.last_name  = ln;

  RETURN total_amount;
END;
$$;

SELECT get_total_amount('AMY', 'LOPEZ')
```
