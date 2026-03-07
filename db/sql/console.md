```bash
# переключится в бд
docker exec -it pg-database psql -U admin -d mydb
```

```bash
\q # выход
\dt # Показать таблицы (tables)
\dn # Показать схемы (namespaces)
\df # Показать функции (functions)
\dv # Показать представления (views)
\du #
\e # открыть редактирование в vim
\l # посмотреть список бд
\c postgres # подключиться к определенной бд
\s # история команд
```
