### WeakMap

ключи в WeakMap должны быть объектами, а не примитивными значениями
если мы используем объект в качестве ключа и если больше нет ссылок на этот объект, то он будет удалён из памяти
(и из объекта WeakMap) автоматически.
не поддерживает перебор и методы keys(), values(), entries(), так что нет способа взять все ключи или
значения из неё.
В WeakMap присутствуют только следующие методы: weakMap.get(key), weakMap.set(key, value), weakMap.delete(key), weakMap.has(key)

```js
let visitCountMap = new Map(); // map: user => число визитов
Function countUser(user){
let count = visitsCountMap.get(user) || 0;
visitCountMap.set(user, count + 1); // Но если объект будет приравнен к нулю, то он останется в памяти,  так как существует еще в Map, если заменить на WeakMap, то при удалении объекта, он также удалится из  коллекции

// Пример для кэширования:
let cache = new WeakMap();
function process(obj) {
  if (!cache.has(obj)) {
    let result = cache.set(obj, result);
  } //вычисляем результат для obj

  return cache.get(obj);
}}
```

### WeakSet

- Она аналогична Set, но мы можем добавлять в WeakSet только объекты (не примитивные значения).
- Объект присутствует в множестве только до тех пор, пока доступен где-то ещё.
- Как и Set, она поддерживает add, has и delete, но не size, keys() и не является перебираемой.

!!! Наиболее значительным ограничением WeakMap и WeakSet является то, что их нельзя перебрать или взять
всё содержимое.
