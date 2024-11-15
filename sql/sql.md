# SQL

## SELECT

Создание таблицы

```sql
CREATE TABLE products ( //создаем таблицу
  id INT NOT NULL, //id, который ниже будет определен, как уникальный идентификатор не может быть нулем
  name STRING, //тип данных
  price MONEY, //тип данных
  PRIMARY KEY(id) //задаем уникальный ID
)
```

Чтение данных из таблицы

```sql
SELECT * FROM products //выбрать всю таблицу, * обозначает выборку всех столбцов
SELECT name, price FROM products //выбрать только несколько столбцов
SELECT * FROM products WHERE id=1 //выбрать ряд с id = 1 из всей таблицы
SELECT count(*) FROM products //считаем сколько строк
SELECT count(*) as cnt FROM products //считаем сколько строк и добавляет столбец cnt
SELECT * FROM `products` WHERE id_cat=1 AND sale>0 //составные выборки
SELECT * FROM `products` WHERE id_cat=1 GROUP BY id_cat //группировка
SELECT * FROM `products` WHERE id_cat IN (SELECT DISTINCT id_cat FROM products WHERE sale > 0)//вложенный запрос

//c сортировкой
SELECT * FROM `products` ORDER BY dt_add //выдаем сортировку по полю
SELECT * FROM `products` ORDER BY dt_add DESC, id_products DESC //выдаем сортировку по двум полям

```

обновление состояния
Вставляем данные

```sql
Вставляем данные
INSERT INTO products
VALUES (2, "Pencil", 0.80, 12)
//выбрать для обновления значения поля price в ячейке с id=2
UPDATE products
SET price = 0.80
WHERE id=2
/добавили столб INT переменной типа число
ALTER TABLE products
ADD stock INT
//удалить
DELETE FROM products
WHERE id=2
```

## JOIN

```sql

SELECT * FROM `products` join cats ON products.id_cat = cats.id_cat или SELECT * FROM `products` join cats USING (id_cat) //выбрать перекрёстную таблицу без произведения таблиц

```

## SQL. FOREIGN

```sql
CREATE TABLE orders (
  id INT NOT NUll,
  order_number INT,
  customer_id INT,
  product_id INT,
  PRIMARY KEY (id), //создаем таблицу
  FOREIGN KEY (customer_id) REFERENCES customers(id), //завязываем поле customer_id в таблице orders с полем id в таблице customers
  FOREIGN KEY (product_id) REFERENCES products(id)
)

JOIN
SELECT orders.order_number, customers.first_name, customers.last_name, customers.adress
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id

```

# MONGODB

## MongoDB. Запуск

Консоль: mongod
Открыть другую вкладку, команда mongo
help:
show dbs - show database names
show collections - show collections in current database
show users - show users in current database
show profile - show most recent system.profile entries with time >= 1ms
show logs - show the accessible logger names
show log [name] - prints out the last segment of log in memory, 'global' is default
use <db_name> - set current database
db.mycoll.find() - list objects in collection mycoll
db.mycoll.find( { a : 1 } ) - list objects in mycoll where a == 1
it - result of the last line evaluated; use to further iterate
DBQuery.shellBatchSize = x - set default number of items to display on shell
exit - quit the mongo shell

## MongoDB. CRUD

```js
db.products.insertOne({_id: 1, name: "Pen", price: 1.20}) //создать запись в DB
db.products.find({name: "Pencil"}) //найти
db.products.find({_id: 1}, {name : 1}) //найти со всеми полямиРезультат: { "_id" : 1, "name" : "Pen" }
db.products.find({_id: 1}, {name : 1, _id: 0}) //найти без отображения некоторых полей
Результат: { "name" : "Pen" }
db.products.updateOne({_id: 1}, {$set: {stock: 32}}) //добавить новое поле stock значение 32
db.products.deleteOne({_id: 2}) //удалить
```

# MONGOOSE

```js
const mongoose = require("mongoose"); //подключение расширения mongoose.connect("mongodb://localhost:27017/fruitsDB")//подключение к БД, если такой нет, то БД создаётся
const fruitSchema = new mongoose.Schema({
  //схема БД
  name: String,
  rating: Number,
  review: String,
});

const Fruit = mongoose.model("Fruit", fruitSchema); //создание БД
const fruit = new Fruit({
  name: "Apple",
  rating: 7,
  review: "Pretty solid as a fruit",
});

fruit.save(); //сохранение БД
const personSchema = new mongoose.Schema({
  //еще одна схема
  name: String,
  age: Number,
});

const Person = mongoose.model("Person", personSchema);
const person = new Person({
  name: "John",
  age: 37,
});

person.save();
```

создаем массив объект добавляемых в БД

```js
const kiwi = new Fruit({
  name: "Kiwi",
  score: 10,
  review: "The best fruit",
});
const orange = new Fruit({
  name: "Orange",
  score: 10,
  review: "Too sour for me",
});
const banana = new Fruit({
  name: "Banana",
  score: 3,
  review: "Wierd texture",
});
//вставка нескольких элементов
Fruit.insertMany([kiwi, orange, banana], function (err) {
  if (err) {
    console.log(err);
  } else {
    //проверка на ошибки
    console.log("Successfully saved all the fruits to fruitsDB");
  }
});
//получение элементов
Fruit.find(function (err, fruits) {
  if (err) {
    console.log(err);
  } else {
    mongoose.connection.close();
    fruits.forEach((item) => {
      console.log(item.name);
    });
  }
});
```
