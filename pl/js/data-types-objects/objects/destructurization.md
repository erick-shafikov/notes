# Деструктуризация объекта

```js
// Справа существующий объект, левая сторона – шаблон
let { var1, var2 } = { var1: "var1", var2: "var2" };

let options = { title: "Menu", width: 100, height: 200 };
let { title, width, height } = options;

// Порядок не имеет значения
let { height, width, title } = options; //тоже самое

// В случае присваивания другой переменной
//из примера выше
let { width: w, height: h, title } = options; //двоеточия показывают что куда идет

// Значения по умолчанию могут быть функциями
let { width = prompt("width"), title = prompt("title") } = options;

// Могут совмещать : и =
let { width: w = 100, height: h = 200, title } = options;

// Взять то, что нужно:
let { title } = options;
```

## Вложенная деструктуризация

```js
let options = {
  size: {
    width: 100,
    height: 200,
  },
  items: ["Cake", "Donut"],
  extra: true,
};

let {
  size: { width, height },
  items: [item1, item2],
  title = "Menu",
} = options;

// size и items отсутствуют так как мы взяли их содержимое
```

## spread

Копирует все enumerable свойства

```js
let options = {
  title: "Menu",
  height: 100,
  width: 200,
};

let { title, ...rest } = options;
alert(rest.height); //100
alert(rest.width); //200

// Подвох с let. JS обрабатывает { } в основном потоке кода как юлок кода
let title, width, height;

const { title, weight, height } = { title: "Menu", width: 200, height: 100 };
//исправленный вариант
({ title, width, height } = { title: "Menu", width: 200, height: 100 });

alert(title);
```
