### Map

Map - это коллекция ключ/значение, как и Object. Но основное отличие в том, что Map позволяет использовать ключи любого типа. Использование объектов в качестве ключей – это одна из известных и часто применяемых возможностей объекта Map. При строковых ключах обычный объект Object может подойти, но для ключей-объектов – уже нет, так как ключами могут быть не только строки, но и объекты

- new Map() - создаёт коллекцию.
- map.set(key, value) записывает по ключу key значение value. ←объект map
- map.get(key) возвращает значение по ключу или undefined, если ключ key отсутствует.
- map.has(key) возвращает true, если ключ key присутствует в коллекции, иначе false.
- map.delete(key) удаляет элемент по ключу key.
- map.clear() очищает коллекцию от всех элементов.
- map.size возвращает текущее количество элементов.

```js
let map = new Map();
map.set("1", "str1");
map.set(1, "num1");
map.set(true, "bool1");

alert(map.get(1)); //num1
alert(map.get("1")); //str1
alert(map.size); //3
```

Объект в качестве ключей

```js
let john = { name: "John" };
let visitCountMap = new Map();
visitCountMap.set(john, 123);
alert(visitCountMap.get(john)); //123
// Вот, что было бы, если бы мы хотели записать в  виде ключа объект в объекте
let john = { name: "John" };
let visitsCountObj = {}; // попробуем  использовать объект
visitsCountObj[john] = 123; // возьмём объект
john как ключ
// Вот как это было записано!
alert( visitsCountObj["[object Object]"] ); //  123

```

#### Перебор

- map.keys() – возвращает итерируемый объект по ключам, map.values() – возвращает итерируемый объект по значениям,
- map.entries() – возвращает итерируемый объект по парам вида [ключ, значение], этот вариант используется
  по умолчанию в for..of.

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

Перебор через forEach

```js
recipeMap.forEach((value, key, map) => {
  alert(`${key}: ${value}`); // огурец: 500 и так  далее
});
```
