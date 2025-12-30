# union

Нужно для объединения двух таблиц по рядам, нужно следить:

- количество столбцов
- за сходимостью по типам
- дубликаты игнорируются

```sql
SELECT column_1, column_2 FROM table1
UNION
SELECT column_1, column_2 FROM table2
```

что бы дубликаты не игнорировались UNION ALL

```sql
-- actor - из какой таблицы, будет в каждой строке указывать на происхождение строки
SELECT first_name, 'actor' FROM actor
UNION ALL
SELECT first_name, 'customer' FROM actor
ORDER BY first_name
```
