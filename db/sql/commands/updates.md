# UPDATE

```sql
UPDATE table_name SET column_name = new_value
```

обычно использую вместе с WHERE

```sql
UPDATE songs SET genre = 'Pop music' WHERE song_id = 4
UPDATE songs SET price=song_id+0.99 --пример с вычислением

```

несколько действий

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

# DELETE

```sql
DELETE FROM table_name WHERE condition -- для всех рядов, для которых condition == true  будут удалены
```

```sql
DELETE FROM songs WHERE song_id IN (3, 4)
```

вернет таблица с удаленными значениями

```sql
DELETE FROM songs WHERE song_id IN (3, 4)
RETURNING song_id -- или *
```
