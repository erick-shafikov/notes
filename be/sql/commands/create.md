# CREATE

Создание таблицы

```sql
CREATE DATABASE myBd; -- создаст БД, за основу будет взята template1
DROP DATABASE myBd --удалить
CREATE DATABASE myBd TEMPLATE template_db; -- создаст БД, за основу будет взята template_db

```

```sql
CREATE TABLE products ( -- создаем таблицу
  id INT NOT NULL, -- id, который ниже будет определен, как уникальный идентификатор не может быть нулем
  name STRING, -- тип данных
  price MONEY, -- тип данных
  PRIMARY KEY(id) -- задаем уникальный ID
)

CREATE DATABASE mybd TEMPLATE meu_template_db -- создать по подобию


```

Создание ролей

```sql
CREATE ROLE joao LOGIN PASSWORD '123456' CREATEDB VALID UNTIL 'infinity';

CREATE ROLE vitor LOGIN PASSWORD '123456' SUPERUSER VALID UNTIL '2030-1-1 00:00';


DROP ROLE vitor -- удалить роль
```
