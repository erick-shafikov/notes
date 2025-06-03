# Map

Map - это коллекция ключ/значение, как и Object. Но основное отличие в том, что Map позволяет использовать ключи любого типа. Использование объектов в качестве ключей – это одна из известных и часто применяемых возможностей объекта Map. При строковых ключах обычный объект Object может подойти, но для ключей-объектов – уже нет, так как ключами могут быть не только строки, но и объекты

```js
// создать можно с помощью new Map и передать entries, если нужны изначальные
const original = new Map([[1, "one"]]);
```

Отличие от объектов:

- у Map нет ключей по умолчанию, в объекта же есть ключи объекта-прототипа
- в объектах ключи либо string либо Symbol, в map ключом может быть и NaN
- в Map ключи располагаются в в порядке добавления, в object - во произвольном
- в Map размер с помощью size в объектах с помощью Object.keys()
- итерация в map с помощью for...of в объекте for...in
- map более оптимизирован для частого добавления ключей
- map напрямую не сериализован

```js
let john = { name: "John" };
let visitCountMap = new Map();

visitCountMap.set(john, 123);
alert(visitCountMap.get(john)); //123

// Вот, что было бы, если бы мы хотели записать в  виде ключа объект в объекте
let john = { name: "John" };
let visitsCountObj = {}; // попробуем  использовать объект
visitsCountObj[john] = 123; // возьмём объект
// john как ключ
// Вот как это было записано!
alert(visitsCountObj["[object Object]"]); //  123
```

Можно добавить ключ как к объекту, но он не будет доступен с помощью методов

```js
const map = new Map();

map["some_key"] = 1;

map.has("some_key"); //false
map("some_key"); //1
```

# методы экземпляра

- new Map() - создаёт коллекцию.
- map.set(key, value) записывает по ключу key значение value. ←объект map

```js
map.get(1); //num1
map.get("1"); //str1
```

- map.get(key) возвращает значение по ключу или undefined, если ключ key отсутствует.

```js
map.set("1", "str1");
map.set(1, "num1");
map.set(true, "bool1");
```

- map.has(key) возвращает true, если ключ key присутствует в коллекции, иначе false.
- map.delete(key) удаляет элемент по ключу key.

```js
//для ключей которых нет, при удалении false
const contacts = new Map();
contacts.set("Jessie", { phone: "213-555-1234", address: "123 N 1st Ave" });
contacts.delete("Raymond"); // false
```

- map.clear() очищает коллекцию от всех элементов.
- map.size возвращает текущее количество элементов.

```js
let map = new Map();

map.get(1); //num1
map.get("1"); //str1
map.size; //3
```

- map.entries()
- map.values()
- map.keys()

# статически методы и свойства

## Map.groupBy()

Позволяет создать map из перебираемого объекта с заданным условием

```js
const inventory = [
  { name: "asparagus", type: "vegetables", quantity: 9 },
  { name: "bananas", type: "fruit", quantity: 5 },
  { name: "goat", type: "meat", quantity: 23 },
  { name: "cherries", type: "fruit", quantity: 12 },
  { name: "fish", type: "meat", quantity: 22 },
];

// ключ для map
const restock = { restock: true };
const sufficient = { restock: false };
const result = Map.groupBy(inventory, ({ quantity }) =>
  quantity < 6 ? restock : sufficient
);
result.get(restock); // [{ name: "bananas", type: "fruit", quantity: 5 }]
result.get(sufficient); //[{name: 'asparagus', type: 'vegetables', quantity: 9}, {name: 'goat', type: 'meat', quantity: 23}, {name: 'cherries', type: 'fruit', quantity: 12}, {name: 'fish', type: 'meat', quantity: 22}]
```

## Symbol.species

Map[Symbol.species] - ссылается на конструктор

```js
Map[Symbol.species]; // function Map()
```

# Перебор

- map.keys() – возвращает итерируемый объект по ключам, map.values() – возвращает итерируемый объект по значениям,
- map.entries() – возвращает итерируемый объект по парам вида [ключ, значение], этот вариант используется
  по умолчанию в for...of.

## for ... of

```js
let recipeMap = new Map([
  ["огурец", 500],
  ["помидор", 350],
  ["лук", 50],
]); // перебор по ключам (овощи)

for (let vegetable of recipeMap.keys()) {
  alert(vegetable); // огурец, помидор, лук
}

// перебор по значениям (числа)
for (let amount of recipeMap.values()) {
  alert(amount); // 500, 350, 50
}
```

Перебор по элементам в формате [ключ, значение]

```js
for (let entry of recipeMap) {
  // то же самое, что и recipeMap.entries()  alert(entry); // огурец,500 (и так далее)
}
```

## forEach

```js
recipeMap.forEach((value, key, map) => {
  alert(`${key}: ${value}`); // огурец: 500 и так  далее
});
```

# Объединение

```js
const first = new Map([
  [1, "one"],
  [2, "two"],
  [3, "three"],
]);

const second = new Map([
  [1, "uno"],
  [2, "dos"],
]);

const merged = new Map([
  // объединение двух map
  ...first,
  ...second,
  //можно добавить массив типа entries
  [1, "un"],
]);
```
