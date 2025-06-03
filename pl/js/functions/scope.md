# область видимости функции

(function scope)

Переменные объявленные внутри функции недоступны вне функции, а переменные объявленные внутри функции имеют доступ к глобальным

```js
// Следующие переменные объявленны в глобальном scope
var num1 = 20,
  num2 = 3,
  name = "Chamahk";

// Эта функция объявленна в глобальном scope
function multiply() {
  return num1 * num2;
}

multiply(); // вернёт 60

// Пример вложенной функции
function getScore() {
  var num1 = 2,
    num2 = 3;

  function add() {
    return name + " scored " + (num1 + num2);
  }

  return add();
}

getScore(); // вернёт "Chamahk scored 5"
```

<!-- Scope и стек функции -->

# Scope и стек функции

стек функции используется для организации вызова самой себя или рекурсии
