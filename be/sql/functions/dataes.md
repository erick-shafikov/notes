# даты

Виды типов даты:

- YYYY-MM-DD - date
- 01:02:03:678 + 02 - time, время c/без временного пояса
- YYYY-MM-DD 01:02:03:678 + 02 - timestamp
- 01:02:03:678 - интервал

# DATE

достает дату

```sql
SELECT *, DATE(date_column) FROM table_name
```

# EXTRACT

для работы с датами используют аргументы

- CENTURY, DAY, DECADE, DOW (day of week), DOY (day of year), EPOCH, HOUR, ISODOW, ISOYEAR, MICROSECONDS, MILLENNIUM, MILLISECONDS, MINUTE, MONTH, QUARTER, SECOND, TIMEZONE, TIMEZONE_HOUR, TIMEZONE_MINUTE, WEEK, YEAR

```sql
EXTRACT (DATE_ARG from date_column) --синтаксис

SELECT
EXTRACT(day from rental_date) as rental_days,
COUNT(*)
FROM rental
GROUP BY rental_days
ORDER BY COUNT(*) DESC
```

## TO_CHAR

Для форматирования даты, можно использовать разные форматы

format: MM-YYYY, YYYY/MM, Day, 'Dy, Month', ...

```sql
TO_CHAR(date_column, format) -- синтаксис

SELECT
EXTRACT(month from payment_date),
TO_CHAR(payment_date, 'Day')
FROM payment
```

## timestamps

```sql
SELECT CURRENT_DATE -- дата
SELECT CURRENT_TIMESTAMP -- Их можно включать в таблицу
```
