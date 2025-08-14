```js
console.log();
console.debug();
console.info();
console.warn();
console.error();
```

```js
const person = { name: "Leslie" };
console.log(person); // Sally будет вычислено в момент просмотра
person.name = "Sally";

const person = { name: "Leslie" };
console.log({ ...person }); //Leslie
person.name = "Sally";
```

```js
console.dir();
console.table();

console.log("Вне группы");

console.group();
console.log("Внутри первой группы");
console.log("Все еще первая группа");
console.groupEnd();

console.group("Название второй группы");
console.log("Внутри второй группы");

console.groupCollapsed();
console.log("Внутри вложенной группы");
console.groupEnd();

console.log("Все еще вторая группа");
console.groupEnd();
```

# timeline

```js
console.time();
slowFunction();
console.timeEnd();
// default: 887.69189453125 ms

console.time("Label");
slowFunction();
console.timeEnd("Label");
// Label: 863.14306640625 ms

console.time("Label");
slowFunctionOne();
console.timeLog("Label");
slowFunctionTwo();
console.timeEnd("Label");
```

# profile

```js
console.profile();
slowFunction();
console.profileEnd();

console.profile("Label");
slowFunction();
console.profileEnd("Label");
```

# misc

```js
const n = 2;
console.assert(n === 1, "Переменная n не равна одному");

const n = 2;
if (n !== 1) console.error("Переменная n не равна одному");

console.clear(); //очистить
```

# counters

```js
console.count();
console.count();
console.countReset();
console.count();
// default: 1
// default: 2
// default: 1

console.count("Label");
console.count("Label");
console.countReset("Label");
console.count("Label");
// Label: 1
// Label: 2
// Label: 1
```

# trace

```js
function firstFn() {
  function secondFn() {
    console.trace();
  }
  secondFn();
}

firstFn();
```
