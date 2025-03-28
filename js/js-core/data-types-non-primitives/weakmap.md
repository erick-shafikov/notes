### WeakMap

- ключи в WeakMap должны быть объектами, а не примитивными значениями
- если мы используем объект в качестве ключа и если больше нет ссылок на этот объект, то он будет удалён из памяти
  (и из объекта WeakMap) автоматически.
- не поддерживает перебор и методы keys(), values(), entries(), так что нет способа взять все ключи или
  значения из неё.
- В WeakMap присутствуют только следующие методы:
- - weakMap.get(key),
- - weakMap.set(key, value),
- - weakMap.delete(key),
- - weakMap.has(key)

Нет перебора элементов

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
