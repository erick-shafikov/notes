# Exceptions

Скрипт не падает, мы получаем возможность обработать ошибку. Синтаксис:

```js
try {
  //код, который выполняется сначала, если в нем нет ошибок, то выполняется до конца try, пропуская catch
} catch (err) {
  //обработка ошибки, если таковая возникает
}

try {
  alert("начало блока");
  notDeclaredVariable;
  alert("Конец блока try"); // Никогда не выполнится
} catch (err) {
  alert("Возникла ошибка");
}
```

!!!Блоки try catch работают только в синтаксически верном коде
!!!Работает синхронно

```js
try {
  setTimeout(function () {
    noSuchVariable; // падение тут
  }, 1000);
} catch (e) {
  alert("не сработает");
}

// блок try catch должен находится внутри функции
setTimeout(function () {
  try {
    noSuchVariable;
  } catch {
    alert("Ошибка поймана");
  }
}, 1000);
```

# Объект ошибки

Для встроенных объектов ошибок два свойства name и message, stack – текущий вызов (строка и т.д.)

```js
try {
  notDeclaredVariable;
} catch (err) {
  alert(err.name); //ReferenceError  alert(err.message); // notDeclaredVariable is not defined
  alert(err.stack); //ReferenceError: notDeclaredVariable is not defined at (стек вызова)
  alert(err); // ошибка целиком
}
```

Можно использовать блок catch без переменной, если нам не нужны подробности

```js
let json = "{ некорректный JSON }";
try {
  let user = JSON.parse(json); //ошибка здесь  alert(user.name); //не сработает
} catch (e) {
  alert("Ошибка");
  alert(e.name);
  alert(e.message);
}
```

## Генерация собственных ошибок

Что если json из предыдущего примера не содержит свойства name

```js
let json = `{ "age": 30 }`; //данные неполные
try {
  let user = JSON.parse(json); //выполняется без ошибки
  alert(user.name); // нет свойства name
} catch (e) {
  alert("Не выполнится");
}
```

Для того, чтобы унифицировать собственную ошибку используется оператор throw. Синтаксис throw <Объект ошибки>

В качестве ошибки может быть что угодно, но желательно, чтобы это был объект со свойствами name и message
Для создания можно использовать Error, SyntaxError, ReferenceError, TypeError и другие
Свойство name – имя конструктора, а message берется из аргумента

```js
let error = new Error(message);

let error = new Error("Ошибка");
alert(error.name); //Error
alert(error.message); //Ошибка
```

## AggregateError

ошибка которая объединяет несколько ошибок

```js
Promise.any([Promise.reject(new Error("some error"))]).catch((e) => {
  console.log(e instanceof AggregateError); // true
  console.log(e.message); // "All Promises rejected"
  console.log(e.name); // "AggregateError"
  console.log(e.errors); // [ Error: "some error" ]
});
```

# throw

Сгенерируем ошибку для нашего примера

throw может пробрасывать любой тип данных

```js
let json = "{'age': 30}";
try {
  let user = JSON.parse(json);

  if (!user.name) {
    throw new SyntaxError("Данные неполные: нет имени");
  }

  alert(user.name); //не выполнится
} catch (e) {
  alert("JSON error" + e.message); //JSON Error
}
```

# rethrow

```js
function readData() {
  let json = "{'age': 30}";
  try {
    let user = JSON.parse(json);
    if (!user.name) {
      throw new SyntaxError("Данные неполные");
    }

    blabla();

    alert(user.name);
  } catch (e) {
    if (e.name == "SyntaxError") {
      alert("JSON Error" + e.name);
    } else {
      throw e;
    }
  }
}

try {
  readData();
} catch (e) {
  alert("Внешний поймал" + e); //Поймали ошибку с blabla()
}
```

# finally

блок finally выполняется после try если не было ошибок и после catch если ошибка была.

```js
try {
 // код
} catch(e) {
 //ловим ошибку
} finally {
 //выполняем всегда
}

try {
alert( "try" );
if (confirm("Сгенерировать ошибку?")) BAD_CODE();
} catch(e) {
alert("catch"); //сработает если согласились сгенерировать ошибку
} finally {
alert(finally); //сработает в любо случае
}

```

## finally и return

Блок finally срабатывает в любом случае выхода из блока try

```js
function func() {
  try {
    return 1;
  } catch (e) {
  } finally {
    alert("finally");
  }
}

alert(func); //сначала "finally" а потом уже 1  try…finally

function func() {
  //начать делать то, что требует измерения
  try {
  } finally {
    // завершить это даже, если упадет
  }
}
```

# Глобальный catch

window.onerror = function(message, url, line, col, error){…};

- message – сообщение об ошибке
- url – url скрипта
- line, col – номер строки и столбца в которых произошла ошибка error – объект ошибки

```html
<script>
  window.onerror = function (message, url, line, col, error) {
    alert(`${message} в ${line} ${col} на ${url}`);
  };

  function readData() {
    badFunc();
  }

  readData(); //Ошибка с описанием
</script>
```

# unhandledrejection

Позволяет отследить не обработанные исключения промиса

```js
window.addEventListener("unhandledrejection", (event) =>
  console.log(event.reasov)
);
```
