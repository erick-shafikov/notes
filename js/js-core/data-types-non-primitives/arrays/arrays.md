# Создание

```js
let arr = Array(1, 2, 3); // [1, 2, 3]
let arr = new Array(); // единственный аргумент – заданное число элементов в массиве, каждый из элементов undefined

let arr = new Array(2); //Создаёт массив длиной 2: [null, null]
// тоже самое что и выше
let arr = [];
arr.length = 2;
let arr = Array(9.3); // RangeError: Invalid array length

// literal syntax
let arr = [];
let arr = [1, , 2]; //1, undefined, 2

let arr = Array.of(1, 2, 3); // Array.from(/* итерируемый объект*/)
let arr = Array.from("string"); //[s, t, r, i, n, g]
```

В массиве могут храниться элементы любого типа:

```js
let arr = [
  " Яблоко",
  { name: "John" },
  true,
  function () {
    alert(" Привет ");
  },
];
```

# доступ к элементам массива

```js
arr[3.4] = "Oranges";
console.log(arr.length); // 0
console.log(arr.hasOwnProperty(3.4)); // true
console.log(arr.0); // синтаксическая ошибка
renderer.3d.setTexture(model, 'character.png');     // синтаксическая ошибка
renderer['3d'].setTexture(model, 'character.png');  // работает как положено

```

# длина массива

уменьшение длины массива приведет к удалению элементов

```js
arr["length"]; //вернет длину

arr.length = 2; //сократить массив до двух элементов, даже если их было больше
```

# Перебор элементов

```js
// Цикл:
let arr = ["Яблоко", "Апельсин", "Груша"];

for (let i = 0; i < arr.length; i++) {
  alert(arr[i]);
}

// С помощью of:
for (let fruit of fruits) {
  //медленнее чем цикл
  alert(fruit);
}

const array = [1, 2, 3];
array.namedKey = 4;
let result = 0;
for (const key in array) {
  result += key;
}
console.log(result); //0012namedKey  for in проходится по всем свойствам array. При этом key всегда является строкой. Поэтому происходит конкатенация строк.
```

# Деструктуризация массива

```js
let arr = ["Ilya", "Kantor"];
let [firstName, surname] = arr; // firstName = arr[0], surname = [1]
alert(surname); //Kantor
alert(firstName); //Ilya
let [fName, name] = "Ilya Kantor".split(" ");

// Пропуск элементов
let [name, , title] = ["Julius", "Cesar", "Consul", "of Romanic Republic"];
alert(title); //Consul
// Работает с любым перебираемым объектом
let [a, b, c] = "abc";
let [one, two, three] = new Set([1, 2, 3]);
// Присваивает что угодно с левой стороны
let user = {};
[user.name, user.surname] = "Ilya Kantor".split(" ");
alert(user.name); // Ilya
// Цикл с .entries()  Для объекта:
let userObj = {
  name: "John",
  age: 30,
};
for (let [key, value] of Object.entries(userObj)) {
  alert(`${key}: ${value}`);
}
// Для коллекции(тоже самое):
let userMap = new Map();
user.set("name", "John");
user.set("age", "30");
for (let [key, value] of userMap) {
  alert(`${key}: ${value}`);
}
```

# Остаточные параметры

```js
let [name1, name2, ...rest] = ["Julius", "Cesar", "Consul", "Roman Republic"];
alert(name1);
alert(name2);
alert(rest[0]); //Consul так как rest является массивом
alert(rest[1]); //Roman Republic
alert(rest.length); //2
```

# Значения по умолчанию

```js
let [firstName, surname] = [];
alert(firstName); //undefined

// Указание значений по	умолчанию. Они могут быть сложнее или даже функциями
// Простые значения по умолчанию:
let [name = "Guest", surname = "Anonyms"] = ["Julius"];
alert(name); //Julius
alert(surname); //Anonyms

// Использование prompt для значений по умолчанию

let [name = prompt("name?"), surname = prompt("surname?")] = ["Julius"];
alert(name); //Julius
alert(surname); //результат prompt
```
