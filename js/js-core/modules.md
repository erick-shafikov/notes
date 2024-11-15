# Модули

Модуль – файл, скрипт. В модулях всегда use strict
export – отмечает переменные и функции, которые должны быть доступны вне текущего модуля import – позволяет импортировать функциональность из других модулей

```js
//sayHI.js
export function sayHI(user) {
  alert("hello ${user}");
}
import { sayHI } from "./sayHI.js";
alert(sayHi); //function
sayHI(John);
```

```html
<!-- main.js -->
<script type = "module">//указываем браузеру, что скрипт является модулем
```

Своя область видимости переменных
Переменные, функции объявленные внутри других модулей не видны в других скрипте. в браузерах существует независимая область видимости для каждого скрипта

```html
<script type="module">
  let user = "John";
</script>
<script type="module">
  alert(user); //error: user is not defined
</script>
```

!!!Путь к модулю должен быть строковым примитивом и не может быть функцией
!!! Мы не можем делать импорт в зависимости от условий или в процессе выполнения

## async, defer

- defer – говорит о том, что браузер должен продолжать загружать страницу и в фоновом режиме загружать скрипт и запустит его. Запускается до DOMContentLoaded разные скрипты будут загружаться по порядку.

- Async скрипты будут загружаться вне зависимости друг от друга и вне зависимости от DOMContentLoaded. Скрипты добавленные через JS ведут себя как async

## Выполнение при импорте

Код в модуле выполняется только один раз при импорте

```js
//alert.js
alert("Модуль выполнен");
//1.js
import "./alert.js"; //модуль выполнен
//2.js
import "./alert.js"; //Ничего не покажет

// задача модуля – инициализация. Если что-то изменится в объекте admin, то другие модули увидят это
//admin.js
export let admin = {
  name: "John",
};
//1.js
import { admin } from "./admin.js";
admin.name = "Pete";
//2.js
import { admin } from "./admin.js";
alert(admin.name); //Pete

// Передача учетных данных в объект admin извне
//admin.js
export let admin = {};
export function sayHi() {
  alert("ready to serve. ${admin.name}");
}
//init.js
import { admin } from "/admin.js";
admin.name = "Pete";
//other.js
import { admin, sayHi } from "./admin.js";
alert(admin.name); //Pete
sayHI(); //Ready to serve, Pete
```

### особенности в браузерах

Модули являются отложенными

- Загрузка внешних модулей не блокирует обработку html
- модули ожидают полной загрузки HTML документа и только затем выполняются
- Скрипты которые идут раньше – выполняются раньше

```html
<script type="module">
  alert(typeof button); //потом этот
</script>

<!-- Сравните с обычным скриптом ниже -->

<script>
  //этот скрипт выполнится первым, обычные скрипты выполняются первыми
  alert(typeof button);
</script>

<button id="button">Кнопка</button>

<!-- Атрибут async работает во встроенных скриптах -->

<script async type="module">
  import { counter } from "./analytics.js";
  counter.count();
</script>
```

## Экспорт

```js
// Экспорт до объявления:
export let month = ["Jan", "Feb"];

export const MODULES_BECAME_STANDARD_YEAR = 2005;

export class User {
  constructor(name) {
    this.name = name;
  }
} //Не ставится точка с запятой после экспорта класс или функции

// Экспорт отдельно от объявления
//say.js
function sayHi(user) {
  alert("Hello ${user}");
}
function syaBye(user) {
  alert("Bye ${user}");
}
export { sayHI, sayBye }; //список для экспорта

// Экспортировать «как»
//say.js
export { sayHi as hi, sayBye as bye }; //теперь hi и bye официальные имена
//main.js
import * as say from "./say.js";
say.hi("John");
say.bye(John);
```

## импорт

```js
import { sayHI, sayBye } from "./main.js/";
sayHi("John");
sayBye("John");

// Но если нужно импортировать много чего, мы можем импортировать все сразу в виде объекта

import * as say from "./say.js"; //недостатки – более длинные имена, преимущества – делаем код более  понятным
say.sayHi("John");
say.sayBye("John");
```

## Импорт «как»

Мы можем использовать as, чтобы импортировать под другими именами

```js
import { sayHi as hi, sayBye as bye } from "./say.js";
hi("John");
bye("John");
```

## экспорт по умолчанию

Модули бывают двух типов: 1. модули с библиотеками функции 2. модули с чем-то одним

```js
export default class User {//в файле может быть только один export default  
    constructor(name){
  this.name = name;
  }}

//main.js
import User from "./user.js"; //не {User}, просто User  
new User("John");

// в силу того, что в файле может быть только один файл export, то можно экспортируемая сущность не  обязана иметь имя

//user.js
export default class{  
    constructor(){}
}
export default function(user){  
    alert();
}
export default [ "item1", "item2", "item3"];
```

так как в файле может быть только один default, то скрипт знает, что мы импортируем

## имя default

В некоторых случаях для обозначения экспорта по умолчанию в качестве имени используется default.
Например, чтобы экспортировать функцию отдельно от ее объявления

```js
function sayHi(user){  
  alert("Hello ${user}");
}
export {sayHi as default}//Тоже самое что и export default перед функцией

// если экспортируется одна сущность по умолчанию и множество именованных
//user.js
export default class User{  
    constructor(name){
this.name = name
}}
export function sayHi(user){  
    alert("Hello ${user}");
}
//main.js
import {default as User, sayHi} from "./user.js";
new User("John");

// И если мы импортируем все как объект import* тогда свойство default как раз и будет экспортом по умолчанию
// Пример 1
import * as user from "./user.js";  
let User = user.default;
new User("John");

// Пример 2 TG
//module.js
export default () => "Hello";
export const name = "World";
// index.js
import * as data from "./module"
console.log(data) // {default: function default(), name: "World"}

Пример 3 TG
//sum.js
export default function sum(x) {
    return x + x;
}
//index.js
import * as sum from '/sum'
//метод вызова функции sum
sum.default(4)

```

## реэкспорт

Синтаксис export … from … позволяет импортировать и сразу экспортировать под другим именем

```js
export { sayHi } from "./say.js"; //реэкспортировать sayHi
export { default as User } from "./user.js"; //реэкспортировать default
```

Структура файлов:

- auth/
- - index.js
- - user.js
- - helpers.js
- - test/
- - - login.js
- - - providers/
- - - github.js facebook.js

мы хотим сделать функциональность нашего пакета доступной через единую точку входа «главный файл», что бы можно было экспортировать import {login, logout} from "auth/index.js". Чтобы внешние разработчики не разбирались со структкрой. Так как функциональность может быть разбросана то мы можем импортировать их в auth.js и тут же экспортировать их наружу

```js
//auth/index.js
import {login, logout} from "./helpers.js";  
export {login, logout};
import User from "./user.js";  
export {User};

// теперь можно писать
import {login} from "auth/index.js".
// Ниже более короткий вариант
export {login, logout} from "./helpers.js";
export {default as User} from "./user.js";

```

## Реэкспорт по умолчанию

```js
//user.js
export default class User{//export User from "./user.js" не сработает
}
// Должно быть:
export {default as User}

// реэкспортирует только именованные экспорты
export * from "./user.js";

// Если мы хотим реэкспортировать именованные \экспорты и экспорты по умолчанию, то нам понадобятся две  инструкции

export * from "./user.js";//для реэкспорта именованных экспортов  export {default} from "./user.js;// для экспорта по умолчанию

```

## Динамические импорты

Выражение import(module) загружает модуль и возвращает промис, результатом которого становится объект модуля, содержащий все его экспорты

```js
let modulePath = prompt("Какой модуль загружать?");  ipmort(modulePath)
.then (obj => {/*Объект модуля*/})
.catch(err => {/*Ошибка модуля*/})

// Внутри асинхронной функции, то можно let module = await import(modulePath)
//say.js
export function hi() {
  alert("Привет");
}

export function bye(){
  alert("Пока");
}
let {hi, bye} = await import("./say.js");
hi();
bye();
// Экспорт по умолчанию
//say.js
export default function() {
alert("Module loaded (export default)!");

let obj = await import("./say.js");
let say = obj.default//
let {default: say} = await import("./say.js");
 say()
}
```

```js
//index.js
console.log("running index.js"); //(2)
import { sum } from "./sum.js";
console.log(sum(1, 2)); //(3) //sum.js
console.log("running sum.js");
export const sum = (a, b) => a + b; //(1)
//вывод в index running sum.js running index.js 3;
```
