# SELECT

Чтение данных из таблицы. Синтаксис

```sql
SELECT column_name FROM table_name -- выбрать колонку из таблицы
SELECT column_name, column_name_1, ... FROM table_name -- выбрать колонки из таблицы
SELECT * FROM table_name -- выбрать все колонки из таблицы
```

# ORDER BY

```sql
SELECT column_name, column_name_1, ... FROM table_name ORDER BY column_name

-- порядок
SELECT column_name, column_name_1, ... FROM table_name ORDER BY column_name ASC -- ASC - в прямом порядке
SELECT column_name, column_name_1, ... FROM table_name ORDER BY column_name DESC -- DESC - в обратном порядке


SELECT column_name_1, column_name_2, ... FROM table_name ORDER BY column_name_1, column_name_2 -- по нескольким колонкам

SELECT column_name_1, column_name_2, ... FROM table_name ORDER BY 1, 2 -- по нескольким c сокращениями (не советуется)
```

# DISTINCT

выбор уникальных элементов по столбцу

```sql
SELECT DISTINCT column_name_1 FROM table_name -- покажет только 1 представителя
SELECT DISTINCT column_name_1, column_name_2 FROM table_name -- покажет уникальные сочетания
```

# LIMIT

добавляется в конце строки

```sql
SELECT DISTINCT column_name_1, column_name_2 FROM table_name LIMIT 100
SELECT column_name, column_name_1, ... FROM table_name ORDER BY column_name LIMIT 100 -- c ORDER BY
```

# AS

алиас - позволяет переименовать колонку

```sql
-- выдаст new_name вместо column_name в выводе
SELECT column_name AS new_name FROM table_name
SELECT COUNT(*) AS custom_name FROM table_name--использование с COUNT
```

# WHERE

Позволяет отфильтровать, операторы: <, >, != (<>), is null, (порядок)

```sql

SELECT column_name1, column_name2 FROM table_name WHERE condition

SELECT * FROM products WHERE id=1 --выбрать ряд с id = 1 из всей таблицы другие операторы <, >, != (<>), is null

-- AND, OR - выполняется первым
SELECT * FROM products WHERE id_cat=1 AND sale>0 --составные выборки
SELECT * FROM products WHERE id_cat=1 OR sale>0

SELECT * FROM products WHERE amount=10.99 OR amount=9.99 AND customer_id = 426 -- SELECT * FROM products WHERE amount=10.99 OR (amount=9.99 AND customer_id) - AND имеет высший приоритет
SELECT * FROM products WHERE (amount=10.99 OR amount=9.99) AND customer_id
SELECT * FROM payment WHERE (
  customer_id = 322
  OR customer_id = 346
  OR customer_id = 354)
AND (amount < 2 OR amount > 10)
ORDER BY customer_id, amount DESC -- выбрать по id, сортировка по id прямая, по amount в обратную
```

## BETWEEN

BETWEEN - включает границы

```sql
SELECT column_1, column_2, ... FROM table_name WHERE column_x BETWEEN a AND b
SELECT column_1, column_2, ... FROM table_name WHERE column_x NOT BETWEEN a AND b
-- даты
SELECT column_1, column_2, ... FROM table_name WHERE column_x NOT BETWEEN 'yyyy-mm-dd hh:mm' AND 'yyyy-mm-dd hh:mm' -- задавать 23:59 для второй даты
```

## IN

выбор из диапазонов

```sql
SELECT column_1, column_2, ... FROM table_name WHERE column_x=x AND column_x=y AND...
SELECT column_1, column_2, ... FROM table_name WHERE column_x IN (x, y, ...)
SELECT column_1, column_2, ... FROM table_name WHERE column_x NOT IN (x, y, ...)
```

## LIKE

найти вхождения в строку

\_ - любой одиночный символ
% - любое повторение:

- - A% - строка начинается с А
- - %A% - А посередине слова

```sql
SELECT column_1, column_2, ... FROM table_name WHERE column_x LIKE 'A%' -- пример найти последовательно регистрозависимое
SELECT column_1, column_2, ... FROM table_name WHERE column_x ILIKE 'A%' -- выключить регистрозависимость
SELECT column_1, column_2, ... FROM table_name WHERE column_x NOT LIKE 'A%'
```

# GROUP BY

```sql
SELECT column_name SUM(column_name) FROM table_name GROUP BY column_name

-- customer_id должно быть в GROUP BY
SELECT customer_id, SUM(amount) FROM payment WHERE customer_id > 3 GROUP BY customer_id

SELECT staff_id, SUM(amount), COUNT(*) AS total FROM payment WHERE amount != 0 GROUP BY staff_id ORDER BY total DESC

-- выборка под двум колонкам
SELECT staff_id, customer_id FROM payment GROUP BY staff_id, customer_id

SELECT staff_id, DATE(payment_date) , SUM(amount), COUNT(*) FROM payment WHERE amount != 0 GROUP BY staff_id, DATE(payment_date) ORDER BY SUM(amount) DESC
```

## GROUPING SETS

сгруппировать по нескольким критериям избегая множественного JOIN

```sql
SELECT
TO_CHAR(payment_date, 'Month') as month,
staff_id,
SUM(amount)
FROM payment
GROUP BY
	GROUPING SETS(
		(staff_id),
		(month),
		(staff_id, month)
	)
ORDER BY 1,2
```

```sql
-- сгруппировать по first_name + last_name + staff_id вывести amount
SELECT
first_name,
last_name,
staff_id,
SUM(amount)
FROM customer c
LEFT JOIN payment p
ON p.customer_id = c.customer_id
GROUP BY
	GROUPING SETS(
		(first_name, last_name),
		(first_name, last_name, staff_id)
	)
```

## ROLLUP

позволяет задать иерархию по фильтрам порядок от column_1 до column_3 в порядке убывания, в формате один ко многим

```sql
GROUP BY
ROLLUP (column_1, column_2, column_3,)

-- эквивалентно
GROUP BY
GROUPING SETS (
  (column_1, column_2, column_3),
  (column_1, column_2),
  (column_1), -- сортировка по column_1
  () -- выбрать всех
)
```

```sql
SELECT
-- Q сокращения для quoter
'Q' || TO_CHAR(payment_date, 'Q') as quater, -- фильтр
EXTRACT(month from payment_date) as month, -- фильтр
DATE(payment_date),
-- агрегирующий показатель
SUM(amount)
FROM payment
GROUP BY
ROLLUP(
  -- приоритет четверть-месяц-день
'Q' || TO_CHAR(payment_date, 'Q'),
EXTRACT(month from payment_date),
DATE(payment_date)
)
ORDER BY 1, 2, 3
-- после каждой четверти будет сумма по четверти
```

## CUBE

позволяет задать все возможные комбинации. Так как могут быть связи между столбцами многие ко многим

```sql
GROUP BY
CUBE (column_1, column_2, column_3,)

-- эквивалентно
GROUP BY
GROUPING SETS (
  (column_1, column_2, column_3),
  (column_1, column_2),
  (column_1, column_3),
  (column_2, column_3),
  (column_1),
  (column_2),
  (column_3),
  ()
)
```

все возможные связи между customer_id, staff_id, payment_date

```sql
SELECT
customer_id,
staff_id,
DATE(payment_date),
SUM(amount)
FROM payment
GROUP BY
CUBE(
customer_id,
staff_id,
DATE(payment_date)
)
ORDER BY 1, 2, 3
-- последняя строка - общая сумма
```

# HAVING

используется для группировки и лимитирования как WHERE только для групп
!!! возможно использоваться только с GROUP BY

```sql
-- выбрать из выборки где сумма amount > 300
SELECT customer_id, SUM(amount) FROM payment GROUP BY customer_id HAVING SUM(amount) > 200

-- несколько условий
SELECT staff_id, DATE(payment_date), SUM(amount), COUNT(*) FROM payment WHERE amount != 0 GROUP BY staff_id, DATE(payment_date) HAVING COUNT(*) = 28 OR COUNT(*) = 29 ORDER BY COUNT(*) DESC

SELECT customer_id, amount, DATE(payment_date), ROUND(AVG(amount),2) AS avg_amount, COUNT(*) FROM payment WHERE DATE(payment_date) IN ('2020-4-28','2020-4-29','2020-4-30') GROUP BY customer_id, amount, DATE(payment_date) HAVING amount > 9 ORDER BY avg_amount DESC
```

````

# comments

```sql
-- single line comment
/*
multiple
line
code
*/
````
