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
