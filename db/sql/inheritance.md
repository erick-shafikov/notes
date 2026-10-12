# Наследование INHERITS

```sql
CREATE TABLE capitals (
  name       text,
  population real,
  elevation  int,    -- (высота в футах)
  state      char(2)
);

CREATE TABLE non_capitals (
  name       text,
  population real,
  elevation  int     -- (высота в футах)
);

CREATE VIEW cities AS
  SELECT name, population, elevation FROM capitals
    UNION
  SELECT name, population, elevation FROM non_capitals;
```

альтернатива с INHERITS

```sql
CREATE TABLE cities (
  name       text,
  population real,
  elevation  int     -- (высота в футах)
);

CREATE TABLE capitals (
  state      char(2) UNIQUE NOT NULL
) INHERITS (cities);
```
