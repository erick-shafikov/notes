# Запуск

```bash
# Консоль:
mongod
# Открыть другую вкладку, команда mongo
help:
show dbs #show database names
show collections # show collections in current database
show users # show users in current database
show profile # show most recent system.profile entries with time >= 1ms
show logs # show the accessible logger names
show log name # prints out the last segment of log in memory, 'global' is default
use db_name # set current database
db.my_coll.find # list objects in collection my_coll
db.my_coll.find( { a : 1 } ) # list objects in my_coll where a == 1
it # result of the last line evaluated; use to further iterate
DBQuery.shellBatchSize = x # set default number of items to display on shell
exit # quit the mongo shell
```

# CRUD

```js
db.products.insertOne({_id: 1, name: "Pen", price: 1.20}) //создать запись в DB
db.products.find({name: "Pencil"}) //найти
db.products.find({_id: 1}, {name : 1}) //найти со всеми полямиРезультат: { "_id" : 1, "name" : "Pen" }
db.products.find({_id: 1}, {name : 1, _id: 0}) //найти без отображения некоторых полей
Результат: { "name" : "Pen" }
db.products.updateOne({_id: 1}, {$set: {stock: 32}}) //добавить новое поле stock значение 32
db.products.deleteOne({_id: 2}) //удалить
```
