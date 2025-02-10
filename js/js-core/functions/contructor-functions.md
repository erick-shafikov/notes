# Construction Function

при создании однотипных объектов можно воспользоваться функцией конструктором через "new"

имя функции-конструктора начинается с большой буквы 2. должна вызываться через оператор new
можно вызывать без скобок, если нет аргументов

```js
function User(name) {
  this.name = name;
  this.isAdmin = false;
}
let user = new User("Вася");
alert(user.name); // Вася  alert(user.isAdmin); // false  происходит следующее:
// Создаётся новый пустой объект, и он присваивается this.
// Выполняется код функции. Обычно он модифицирует this, добавляет туда новые свойства.
// Возвращается значение this. 4. При вызове return с объектом, будет возвращён объект, а не this

function User(name) {
  //this = {}; (неявно)
  //добавляет к this  this.name = name;  this.isAdmin = false;
  //return this(неявно)
}
// Любая функция может быть использована как конструктор

new (function () {})();
let user = new (function () {
  this.name = "John";
  this.isAdmin = false;
  //	и т.д. такой конструктор вызывается один раз
})();
```

Используя специальное свойство new.target внутри функции мы можем проверить вызвана ли функция при помощи опреатора new или без него, если да, то в new.target будет сама функиця, в противном случае undefined

```js
function User() {
  alert(new.target);
}
User(); //undefined
new User(); // код User

// функцию можно вызывать как с new так и без него
function User(name) {
  if (!new.target) {
    return new User(name);
  }

  this.name = "name";
}

let john = User("John");
alert(john.name); // John

// Без new можно войти в заблуждение по поводу создания объекта
```

задача конструкторов – записать все необходимое в this
при вызове return с объектом будет возвращен объект а не this при вызове return с примитивным значением, оно будет отброшено

```js
function BigUser() {
  this.name = "Вася";
  return { name: "Godzilla" }; // <— возвращает этот объект
}

alert(new BigUser().name); // Godzilla, получили этот объект

function SmallUser() {
  this.name = "Вася";
  return; // <— возвращает this
}
alert(new SmallUser().name); // Вася
// При вызове return с примитивности значением , примитивное значение - отбросится
```

## Методы в конструкторе

```js
function User(name) {
  this.name = name;
  this.sayHi = function () {
    alert("Меня зовут: " + this.name);
  };
}
let vasya = new User("Вася");
vasya.sayHi();
// Меня зовут: Вася /* vasya = { name: "Вася", sayHi: function() { ... } }*/
```

Если переменную во внутреннем лексическом окружении мы не можем изменить извне, то это
можно сделать с помощью вложенных функций или свойств

```js
// TG Передача по ссылке
class Counter {
  constructor() {
    this.count = 0;
  }
  increment() {
    this.count++;
  }
}
const counterOne = new Counter();
counterOne.increment();
counterOne.increment();
const counterTwo = counterOne; //создается ссылка на тот же самый объект
counterTwo.increment(); //увеличиваем еще на один
console.log(counterOne.count); //3
```
