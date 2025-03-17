# Iterable objects

- Итерируемые объекты – это объекты, которые ревизуют метод [Symbol.iterator]
- Объекты String, Array, TypedArray, Map и Set являются итерируемыми, потому что их прототипы содержат метод Symbol.iterator.
- Псевдо массивы – это объекты у которых есть индексы и свойство length Возможны объекты смешанного типа
- Итерируемый объект не может быть псевдо массивом и псевдо массив не может быть итерируемым объектом

```js
//псевдо массив, но его нельзя итерировать, есть индексы и length -> псевдо массив
let arrayLike = {
  0: "Hello",
  1: "World",
  length: 2,
};

for (let item of arrayLike) {
  //
}
```

- Когда цикл for..of запускается, он вызывает этот метод один раз (или выдаёт ошибку, если метод не найден). Этот метод должен вернуть итератор объект с методом next.
- Дальше for..of работает только с этим возвращённым объектом.
- Когда for..of хочет получить следующее значение, он вызывает метод next() этого объекта.
- Результат вызова next() должен иметь вид {done: Boolean, value: any}, где done=true означает, что итерация закончена, в противном случае value содержит очередное значение.

```js
let range = { from: 1, to: 5 }; // 1. вызов for..of сначала вызывает эту функцию
range[Symbol.iterator] = function () {
  // ...она возвращает объект итератора:
  // 2. Далее, for..of работает только
  // с этим итератором, запрашивая у него новые значения
  return {
    current: this.from,
    last: this.to,
    // 3. next() вызывается на каждой итерации цикла for..of
    next() {
      // 4. он должен вернуть
      // значение в виде объекта {done:.., value :...}
      if (this.current <= this.last) {
        return { done: false, value: this.current++ };
      } else {
        return { done: true };
      }
    },
  };
}; // теперь работает!
for (let num of range) {
  alert(num); // 1, затем 2, 3, 4, 5
}
```

```js
for (let char of "test") {
  // срабатывает 4 раза: по одному для каждого символа
  alert(char); // t, затем e, затем s, затем t
}
```

# Iterator

Объект Iterator— это объект, который соответствует протоколу итератора (метод next() и {done: true, value: ''}). Все итерируемые объекты наследуются от Iterator-объекта, который предоставляет методы

```js
const nameToDeposit = new Map([
  ["Anne", 1000],
  ["Bert", 1500],
  ["Carl", 2000],
]);
//вместо двойного прохода
const totalDeposit = [...nameToDeposit.values()].reduce((a, b) => a + b);
//можно обойтись одним
const totalDeposit = nameToDeposit.values().reduce((a, b) => a + b);
```

## Статические методы:

- Iterator.from()

## Свойства экземпляра:

- Iterator.prototype[Symbol.toStringTag]

## методы экземпляра (нет в safari):

### drop() - пропускает несколько итераций

```js
// fibonacci - ф-цф реализует генератором числа фибоначчи
const seq = fibonacci().drop(2);
console.log(seq.next().value); // 2
console.log(seq.next().value); // 3
```

### every() - проверка элементов итераций на условие

```js
const isEven = (x) => x % 2 === 0;
console.log(fibonacci().every(isEven)); // false

const isPositive = (x) => x > 0;
console.log(fibonacci().take(10).every(isPositive)); // true
console.log(fibonacci().every(isPositive)); // Never completes
```

### filter()

```js
const seq = fibonacci().filter((x) => x % 2 === 0);
console.log(seq.next().value); // 2
console.log(seq.next().value); // 8
console.log(seq.next().value); // 34
```

### find()

```js
const isEven = (x) => x % 2 === 0;
console.log(fibonacci().find(isEven)); // 2

const isNegative = (x) => x < 0;
console.log(fibonacci().take(10).find(isNegative)); // undefined
```

- flatMap() - запускает функцию для каждого элемента

```js
const map1 = new Map([
  ["a", 1],
  ["b", 2],
  ["c", 3],
]);
const map2 = new Map([
  ["d", 4],
  ["e", 5],
  ["f", 6],
]);

const merged = new Map([map1, map2].values().flatMap((x) => x));
console.log(merged.get("a")); // 1
console.log(merged.get("e")); // 5
```

### forEach()

```js
new Set([1, 2, 3]).values().forEach((v) => console.log(v));
```

### map()

```js
const seq = fibonacci().map((x) => x ** 2);
console.log(seq.next().value); // 1
console.log(seq.next().value); // 1
console.log(seq.next().value); // 4
```

- reduce()
- some()
- take()
- toArray()
- [Symbol.iterator]()

# BPs

# числа Фибоначчи

Фибоначчи

```js
function* fibonacci() {
  let current = 1;
  let next = 1;
  while (true) {
    yield current;
    [current, next] = [next, current + next];
  }
}
```
