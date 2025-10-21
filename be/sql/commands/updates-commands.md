# UPDATE

```sql
UPDATE pg_database SET datistemplate = TRUE WHERE datname = 'meubd' --сделает базу шаблонной
UPDATE pg_database SET datistemplate = FALSE WHERE datname = 'meubd' -- После этого база meubd снова обычная — её можно удалить и использовать как обычную базу данных.
```

# комбинации

обновление состояния
Вставляем данные

```sql
-- Вставляем данные
INSERT INTO products
VALUES (2, "Pencil", 0.80, 12)
-- выбрать для обновления значения поля price в ячейке с id=2
UPDATE products
SET price = 0.80
WHERE id=2
-- добавили столб INT переменной типа число
ALTER TABLE products
ADD stock INT
-- удалить
DELETE FROM products
WHERE id=2
```
