# JOIN

```sql

SELECT * FROM `products` join cats ON products.id_cat = cats.id_cat или SELECT * FROM `products` join cats USING (id_cat)
-- --выбрать перекрёстную таблицу без произведения таблиц

```
