# TS и DOM

В отличие от JavaScript, TypeScript не имеет доступа к DOM. Это означает, что при обращении к DOM-элементам TypeScript не может быть уверен в том, что они существуют.

```ts
const link = document.querySelector("a");
console.log(link.href); //Ошибка: возможно объект является 'null'. TypeScript не может быть уверен в его существовании, т.к. у него нет доступа к DOM
// Здесь мы говорим TypeScript, что эта ссылка существует
const link = document.querySelector("a")!;
console.log(link.href); // habr.com
```

Обратите внимание, что нам не нужно объявлять тип переменной link . Как мы уже знаем, TypeScript сам понимает (с помощью определения типа), что эта переменная типа HTMLAnchorElement .

Но что, если нам надо найти DOM-элемент по его классу или id? TypeScript не может определит тип такой переменной, потому что она может быть чем угодно.

```ts
const form = document.getElementById('signup-form');
console.log(form.method);
// ОШИБКА: Возможно, объект является 'null'.
// ОШИБКА: У типа 'HTMLElement' нет свойства 'method'.
надо сообщить TypeScript, что мы уверены в том, что этот элемент существует, и что он типа HTMLFormElement Для этого используется приведение типов (ключевое слово as):
const form = document.getElementById('signup-form') as HTMLFormElement;
console.log(form.method); // post

```

## для событий

```ts
const form = document.getElementById("signup-form") as HTMLFormElement;
form.addEventListener("submit", (e: Event) => {
  console.log(e.target); // ОШИБКА: Свойство 'target' не существует у типа 'Event'. Может, вы имели ввиду 'target'?
});
```
