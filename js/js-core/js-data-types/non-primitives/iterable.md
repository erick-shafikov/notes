<!-- Iterable objects --------------------------------------------------------------------------------->

### Iterable objects

Итерируемые объекты – это объекты, которые ревизуют метод Symbol.iterator Псевдо массивы – это объекты у которых есть индексы и свойство length Возможны объекты смешанного типа Итерируемый объект не может быть псевдо массивом и псевдо массив не может быть итерируемым объектом

```js
let arrayLike = {//псевдо массив, но его нельзя итерировать, есть индексы и length -> псевдо массив
0: "Hello",
1: "World",
length: 2
};

for (let item of arrayLike)

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

### linked list

```js
let list = {
  value: 1,
  next: {
    value: 2,
    next: {
      value: 3,
      next: {
        value: 4,
        next: null,
      },
    },
  },
};
// Разделить связанный список
let secondList = list.next.next;
list.next.next = null;
// Обледенить:
list.next.next = secondList

// для добавления нового:
list = {"new item", next: list};

// Вывод по порядку(цикл):
function printList(list) {
let tmp = list;
while (tmp) {
  alert(tmp.value);
  tmp = tmp.next;
}}
// Вывод по порядку(рекурсия):
function printList(list) {
alert(list.value); // выводим
// текущий элемент
if (list.next) {  printList(list.next); //делаем то же самое для остальной  части списка
}}
// Вывод в обратном(рекурсия):
function printReverseList(list) {  if (list.next) {
printReverseList(list.next);
}
alert(list.value);
}
if(obj.next != null){
revPrintList(obj.next);  alert(obj.value);
} else {
alert(obj.value);
}

```
