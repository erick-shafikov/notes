Логические преобразования

```js
0, null, undefined, NaN, ""; //false
// все остальные true
```

```js
let value = Boolean(value); // преобразовывает все в boolean

var b = new Boolean(false);
if (b) // это условие true
if (b == true) // это условие false

```

# new Boolean()

```js
var x = new Boolean(false);
if (x) {
  // этот код будет выполнен
}

var x = false;
if (x) {
  // этот код не будет выполнен
}

var myFalse = new Boolean(false); // начальное значение равно false
var g = new Boolean(myFalse); // начальное значение равно true
var myString = new String("Привет"); // строковый объект
var s = new Boolean(myString); // начальное значение равно true
```

null/undefined не имеют методов
