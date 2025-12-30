# CAST

позволяет приводить типы

```sql
SELECT
COALESCE(CAST(actual_arrival-scheduled_arrival AS VARCHAR), 'not arrived'),
CAST(ticket_no AS bigint)
FROM flights
```

# COALESCE

вернет первое не-null значение, позволяет подставить fallback значение

```sql
-- должны сходится по типу
-- в примере actual_arrival и scheduled_arrival даты
SELECT
COALESCE(actual_arrival-scheduled_arrival, '0:00')
FROM flights
```
