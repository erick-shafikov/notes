Создание пользователя

```sql
CREATE USER user_name WITH PASSWORD 'pwd123'
```

Есть пользователи и есть роли, одной роли может соответствует несколько пользователей

Привилегии:

- select
- insert
- update
- delete
- truncate
- usage - видеть объекты схемы
- create
- connect
- execute - для функция и схем
- usage

ROLE = USER + login

!!! только root и владелец может раздавать роли

```sql
-- Создание ролей
CREATE ROLE joao LOGIN PASSWORD '123456' CREATEDB VALID UNTIL 'infinity';
CREATE ROLE vitor LOGIN PASSWORD '123456' SUPERUSER VALID UNTIL '2030-1-1 00:00';
DROP ROLE vitor -- удалить роль

-- Создание привилегий
GRANT SELECT ON customer TO user_name -- конкретная таблица
GRANT SELECT ON ALL TABLES IN SCHEMA TO user_name --все
GRANT SELECT ON ALL TABLES IN SCHEMA TO user_name WITH GRANT OPTION
-- отзыв
REVOKE privilege
ON db_object
FROM USER | ROLE | PUBLIC

REVOKE privilege
ON db_object
FROM USER | ROLE | PUBLIC
GRANTED BY USER | ROLE

REVOKE INSERT ON table_name FROM user_name
REVOKE ALL PRIVILEGES ON table_bane FROM PUBLIC
```

```sql
-- позволить создавать таблицы
LATER USER user_name CREATEDB
```

```sql
-- позволить создавать таблицы
ALTER USER user_name CREATEDB
```

```sql
-- передача ролей
GRANT user_name_1 TO user_name_2
GRANT role_name TO user_name_2
```

# TABLESPACE

```sql
CREATE TABLESPACE secondary LOCATION '/var/lib/postgresql/13/'; -- Создаёт таблспейс с именем secondary LOCATION — путь к папке, где физически будут храниться данные.
ALTER DATABASE meudb_admin SET TABLESPACE secondary; -- Перемещает всю базу данных meudb_admin в таблспейс secondary.
ALTER TABLE tabela1 SET TABLESPACE secondary; -- Перемещает конкретную таблицу tabela1 в таблспейс secondary
```
