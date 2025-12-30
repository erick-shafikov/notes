# Работа со строками

# UPPER, LOWER, LENGTH

```sql
SELECT
UPPER(email) AS email_upper, -- uppercase
LOWER(email) AS email_lower, -- lowercase
LENGTH(email) as email_length, -- длина строки
email as email_as_it_is
FROM customer
WHERE LENGTH(email) < 30 -- для фильтрации
```

# LEFT, RIGHT

```sql
SELECT
LEFT(first_name, 3), -- третий символ сначала
RIGHT(first_name, 3), -- третий символ с конца
RIGHT((LEFT(first_name, 2)),1) -- костыль для получения 2 символа
FROM customer
```

substring

получить подстроку

```sql
SUBSTRING(string from start  [for length]) --синтаксис
SUBSTRING(email from POSITION('.' in email)  [for length]) -- вместе с Position
```

# конкатенация

|| - знак конкатенации

```sql
SELECT
LEFT(first_name, 1) || '. ' || LEFT(last_name, 1)|| '.'
AS initials
FROM customer
```

# позиция символов в строке

позволяет получить номер, на которой находится символ

```sql
SELECT
POSITION('@' IN email), -- достать номер символа '@'
LEFT(email, POSITION('@' IN email) - 1), -- достать подстроку до символа
LEFT(email, POSITION(last_name IN email)), -- можно проверить на вхождение определенного столбца в строку и получить. Получим индекс с которого НАЧИНАЕТСЯ вхождение
FROM customer
```

# REPLACE

```sql
SELECT
CAST(REPLACE(passenger_id, ' ', '') AS BIGINT)
FROM tickets
```
