# Transaction

единица действия в таблице

```sql
BEGIN;

OPERATION1;
OPERATION2;

-- Все действия должны быть закомиченны, что бы изменения были видны всем пользователям

COMMIT
```

```sql
BEGIN;

UPDATE acc_balance
SET amount = amount - 100
WHERE id=1;

UPDATE acc_balance
SET amount = amount - 100
WHERE id=1;

COMMIT;
```

# ROLLBACK

```sql
BEGIN;

OPERATION1;
OPERATION2;
OPERATION3;
OPERATION4;

ROLLBACK;
COMMIT;

```

```sql
BEGIN;

OPERATION1;
OPERATION2;
-- все до этого применится
SAVEPOINT op2
OPERATION3;
OPERATION4;
SAVEPOINT op2;
-- все ниже откатится

ROLLBACK TO SAVEPOINT op3; -- закончить транзакцию
RELEASE SAVEPOINT op3; -- не заканчивает операции
COMMIT;

```
