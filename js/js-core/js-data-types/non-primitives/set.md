### Set

это особый вид коллекции: «множество» значений БЕЗ КЛЮЧЕЙ, где каждое значение может появляться только один раз. Его основные методы это:

(обычно это массив), то копирует его значения в новый Set.

- set.add(value) добавляет значение (если оно уже есть, то ничего не делает), возвращает тот же объект set.
- set.delete(value) удаляет значение, возвращает true если value было в множестве на момент вызова, иначе false.
- set.has(value) возвращает true, если значение присутствует в множестве, иначе false. set.clear() удаляет все имеющиеся значения.
- set.size возвращает количество элементов в множестве.
- Set.values(), set.keys(), set.entries() – все эти методы имеет и Set

```js
let set = new Set();
let john = { name: "John" };
let pete = { name: "Pete" };
let mary = { name: "Mary" };
Set.add(john);
Set.add(john);
Set.add(mary);
Set.add(pete);
Set.add(pete);
alert(set.size); //3  For(let user of set){  Alert(user.name);}

// Можно с помощью for…of или forEach
let set = new Set(["1", "2", 3]);
for (let value of set) {
  alert(value);
}
set.forEach((value, valueAgain, set) => {
  // 3 аргумента для совместимости с Map
  alert(value);
});
```
