# CTE

замена целых запросов

```sql
WITH cte_name AS (
  -- cte запрос
  SELECT column_1, column_2, ....
  FROM table_name
  WHERE condition
)
SELECT column_1, column_2,...
FROM cte_name
WHERE condition
```

использование cte в cte

```sql
cte2 AS (
    -- cte запрос
  SELECT column_1, column_2, ....
  FROM cte_name
  WHERE condition
)
```

# recursive

рекурсивные

```sql
WITH RECURSIVE cte_name AS (
  -- anchor member
  SELECT columns
  FROM table
  WHERE condition

  UNION ALL
  -- recursive member
  SELECT columns
  FROM table_name
  WHERE condition
)

SELECT columns
FROM cte_name
```

```sql
-- пример ряды от 1 до 5
WITH RECURSIVE count_cte AS (
  -- anchor member
  SELECT 1 AS number

  UNION ALL

  -- recursive member
  SELECT number + 1
  FROM count_cte
  WHERE number < 5
)

SELECT number
FROM count_cte
```
