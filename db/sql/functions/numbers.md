# aggregation functions

## COUNT

!!! null не будут учитываться

```sql
SELECT COUNT(*) FROM table_name
SELECT COUNT(DISTINCT *) FROM table_name
SELECT COUNT(column_name) FROM table_name
SELECT count(*) as cnt FROM products --считаем сколько строк и добавляет столбец cnt
```

## SUM, AVG, MIN, MAX, ROUND

- !!!нельзя использовать несколько колонок
- у ROUND есть второй аргумент - до скольки знаков после запятой нудно округлять

```sql
SELECT SUM(column_name) FROM table_name
-- несколько
SELECT SUM(amount), ROUND(AVG(amount),3) FROM payment
```
