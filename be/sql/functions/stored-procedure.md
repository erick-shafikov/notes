# STORE PROCEDURE

- в функциях нельзя использовать транзакции
- функция ничего не возвращает

```sql
CREATE PROCEDURE procedure_name (parma_1,...)
  LANGUAGE plpgsql
AS
$$
DECLARE
<variables>
BEGIN
<procedures>
END;
$$
```

```sql
CREATE PROCEDURE sp_transfer
(tr_amount INT, sender INT, recipient INT)
	LANGUAGE plpgsql
AS
$$
BEGIN
-- subtract from sender's balance
UPDATE acc_balance
SET amount = amount - tr_amount
WHERE id=sender;

-- subtract from sender's balance
UPDATE acc_balance
SET amount = amount + tr_amount
WHERE id=recipient;

COMMIT;
END;
$$

CALL sp_transfer(500, 1, 2)

SELECT * FROM acc_balance
```
