# data types

Типы данных:

- Числовые типы (Numeric Types):

  - Целые числа:
    - SMALLINT — малые целые 2 байта
    - INT / INTEGER — стандартные целые 4 байта
    - BIGINT — большие целые 8 байт
    - TINYINT (MySQL) — очень маленькие целые
    - SERIAL / BIGSERIAL (PostgreSQL) — автоинкремент (variable 1 to 2147483647 для id)
  - Числа с плавающей точкой:
    - REAL — float
    - DOUBLE PRECISION — double
    - FLOAT(p) — плавающая точность
  - Десятичные числа (фиксированная точность):
    - NUMERIC(p, s) — точное число 4 байта p - всего цифр, s - послед запятой
    - DECIMAL(p, s) — аналог numeric

- Строковые типы (Character Types):

  - Текстовые строки:
    - CHAR(n) — фиксированная длина
    - VARCHAR(n) — ограниченная длина
    - TEXT — неограниченный текст
  - Бинарные строки:
    - BYTEA (PostgreSQL) — бинарные данные
    - BLOB (MySQL/SQLite) — большие бинарные объекты
    - VARBINARY(n) (MySQL) — бинарные данные с ограничением

- Дата и время:

  - DATE — дата '2025-12-01'
  - TIME — время '01:02:03.678'
  - TIME WITH TIME ZONE — время с TZ
  - TIMESTAMP — дата и время '2025-12-01 01:02:03.678' (date + time)
  - TIMESTAMP WITH TIME ZONE — дата, время, TZ
  - INTERVAL (PostgreSQL) — промежутки времени '3 days 01:02:03.678'

- Логические типы:

  - BOOLEAN / BOOL — логические значения

- Идентификаторы:

  - UUID — универсальный идентификатор
  - SERIAL / BIGSERIAL — автоинкремент
  - AUTO_INCREMENT (MySQL)

- Денежные типы:

  - MONEY (PostgreSQL)
  - DECIMAL / NUMERIC — рекомендуется использовать

- Массивы (PostgreSQL) {'xxxx', 'yyyy'}:

  - integer[]
  - text[]
  - uuid[]
  - массивы произвольных типов

- Документные типы:

  - JSON — хранение json
  - JSONB — бинарный json (PostgreSQL)
  - XML — xml-документы

- Геометрические и пространственные типы:

  - PostGIS / PostgreSQL:
    - GEOMETRY
    - GEOGRAPHY
    - POINT
    - LINESTRING
    - POLYGON
  - MySQL Spatial:
    - POINT
    - MULTIPOINT
    - LINESTRING
    - POLYGON
    - другие геотипы

- Специальные типы:

  - PostgreSQL:
    - ENUM — перечисления ENUM ('G', 'PG', ...)
    - CIDR — подсети
    - INET — IP-адреса
    - MACADDR — MAC-адреса
    - HSTORE — key-value
    - TSVECTOR — для полнотекстового поиска
    - TSQUERY — запросы поиска
    - OID — системный идентификатор
  - MySQL:
    - ENUM
    - SET
    - YEAR
  - SQLite (динамическая типизация):
    - TEXT
    - INTEGER
    - REAL
    - BLOB
    - NULL

- Системные типы:
  - SERIAL / BIGSERIAL — автоинкрементные идентификаторы
  - OID — системный внутренний ID (PostgreSQL)
