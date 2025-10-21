# SELECT

Чтение данных из таблицы

```sql
SELECT * FROM products --выбрать всю таблицу, * обозначает выборку всех столбцов
SELECT name, price FROM products --выбрать только несколько столбцов
SELECT * FROM products WHERE id=1 --выбрать ряд с id = 1 из всей таблицы
SELECT count(*) FROM products --считаем сколько строк
SELECT count(*) as cnt FROM products --считаем сколько строк и добавляет столбец cnt
SELECT * FROM `products` WHERE id_cat=1 AND sale>0 --составные выборки
SELECT * FROM `products` WHERE id_cat=1 GROUP BY id_cat --группировка
SELECT * FROM `products` WHERE id_cat IN (SELECT DISTINCT id_cat FROM products WHERE sale > 0)--вложенный запрос

-- c сортировкой
SELECT * FROM `products` ORDER BY dt_add --выдаем сортировку по полю
SELECT * FROM `products` ORDER BY dt_add DESC, id_products DESC --выдаем сортировку по двум полям

```
