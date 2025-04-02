# STORAGE. ХРАНЕНИЕ ДАННЫХ В БРАУЗЕРЕ

## local и session storage

отличие от cookie – не отправляют запрос на сервер
сервер не может манипулировать через HTTP-заголовки

setItem(key, value) – сохранить пару ключ/значение
getItem(key) – получить данные по ключу key
removeItem(key) – удалить данные с ключом key
clear() – удалить все
key(index) – получить ключ на заданной позиции
length – количество элементов в хранилище

### Демо localStorage

это объект один на все вкладки и окна в рамках одного источника
данные не имеют срока давности. по которому стираются, сохраняются после перезапуска браузер или ОС
когда мы модифицируем данные срабатывает событие storage

Доступ
как у обычных объектов

```js
localStorage.setItem("test", 1); //добавить item в LS
alert(localStorage.getItem("test")); //1
localStorage.test = 2; //можно добавлять свойства как у обычных объектов
alert(localStoragel.test); //получить доступ - не рекомендуется
delete localStorage.test; // получить доступа - не рекомендуется
```

НО

```js
localStorage.setItem("toString", 1); //сработает
localStorage.setItem("length", 1); //сработает
localStorage.length = 1; //не сработает
localStorage.toString = 1; //не сработает
```

### Перебор ключей

```js
//проход как по обычным массивам
for (let i = 0; i < localStorage.length; i++) {
  let key = local.localStorage.key(i); //ключи доступны по индексам
  alert(localStorage.getItem(key));
} //как по обычному объекту
for (let key in localStorage) {
  alert(key); //покажет и setItem и getItem и другие встроенные свойства
} //отфильтруем собственные свойства
for (let key in localStorage) {
  if (!localStorage.hasOwnProperty(key)) {
    continue;
  }
  alert(localStorage.getItem(key));
} //получим собственные ключи
let keys = Object.keys(localStorage);
for (let key in keys) {
  alert(key);
}
```

## Только строки

ключ и значение - только строки, если использовать другой тип - преобразуется к строке

```js
sessionStorage.user = { name: "John" };
alert(sessionStorage.user); //[object Object]
// можно использовать JSON
sessionStorage.user = JSON.stringify({ name: "John" });
let user = JSON.parse(sessionStorage.user);
alert(user.name); //John
//для JSON.stringify добавлены параметры форматирования, чтобы объект выглядел лучше
alert(JSON.stringify(localStorage, null, 2));
```

## sessionStorage

существует только в рамках текущей вкладки, другая вкладка будет иметь другое хранилище
разделяется между фреймами, если они из одного и того же источника

```js
sessionStorage.setItem("test", 1);
alert(sessionStorage.getItem("test"));
```

но если открыть в другой вкладке – данные не будут доступны (null)

## Событие storage

При обновлении данных в localStorage или sessionStorage генерируется событие со свойствами, срабатывает при вызове setItem, removeItem, clear

key – ключ который обновился(null, если был вызван .clear())
oldValue – старое значение (null, если ключ только добавлен)
newValue – новое значение (null, если ключ был уделен)
url – url документа, \где произошло событие
storageArea – объект localStorage или sessionStorage

если открыть страницу в двух разных браузерах, то каждое из них будет реагировать на обновление

```js
window.onstorage = (event) => {
  if (event.key != "now") return;
  alert(event.key + ":" + event.newValue + " at " + event.url);
};
localStorage.setItem("now", Date.now());
```

## INDEXDB

встроенная база данных, более мощная чем localStorage

доступно несколько типов ключей, значения могут быть любыми
Поддерживает транзакции
Поддерживает запросы в диапазоне ключей и индексы
размер больше чем у LS

Открытие базы данных

```js
//name – название БД, version – версия БД
let openRequest = indexedDB.open(name, version);
```

Назначить обработчик объекта БД openRequest

success – БД готова, готов объект openRequest.result
error – не удалось открыть БД
upgradeneeded – БД открыта, но ее схема устарела

```js
// открытие БД
let openRequest = indexedDB.open("store", 1);
openRequest.onupgradeneeded = function () {
  //срабатывает, если на клиенте нет БД и выполняет инициализацию
};
openRequest.onerror = function () {
  console.log("error", openRequest.error);
};
openRequest.onsuccess = function () {
  let db = openRequest.result; //продолжить работу с базой данных
};

// открываем вторую БД
let openRequest = indexedDB.open("store", 2);
// проверить существование указанной версии базы данных, обновить по мере необходимости:
openRequest.onupgradeneeded = function (event) {
  // версия существующей базы данных меньше 2 (или база данных не существует)
  let db = openRequest.result;
  switch (
    event.oldVersion // существующая (старая) версия базы данных
  ) {
    case 0: // версия 0 означает, что на клиенте нет базы данных // выполнить инициализацию
    case 1: // на клиенте версия базы данных 1// обновить
  }
};

// удаление БД
let deleteRequest = indexedDB.deleteDatabase(name); // deleteRequest.onsuccess/onerror отслеживает результат
```

### INDEXDB. Проблема параллельного обновления

```js
let openRequest = indexedDB.open("store", 2);
openRequest.onupgradeneeded = () => {};
openRequest.onerror = () => {};
openRequest.onsuccess = function () {
  let db = openRequest.result;
  db.onversionchange = function () {
    //при попытке обновление на базе данных срабатывает событие onversionchange
    db.close();
    alert("База данных устарела, пожалуйста, перезагрузите страницу.");
  }; // ...база данных доступна как объект db...
};
openRequest.onblocked = function () {
  // есть другое соединение к той же базе
  // и оно не было закрыто после срабатывания на нём db.onversionchange
};
```

### INDEXDB. Хранилище объектов

можем хранить любые типы данных, в том числе и объекты
Ключ должен быть одним из следующих типов: number, date, string, binary или array. Это уникальный идентификатор: по ключу мы можем искать/удалять/обновлять значения.

Создание хранилища объектов
db.createObjectStore(name[, keyOptions]);
операция является синхронной

- **name** – это название хранилища, например "books" для книг,
- **keyOptions** – это необязательный объект с одним или двумя свойствами:
- **keyPath** – путь к свойству объекта, которое IndexedDB будет использовать в качестве ключа, например id.autoIncrement – если true, то ключ будет формироваться автоматически для новых объектов, как постоянно увеличивающееся число.

Удаление хранилища объектов

```js
db.deleteObjectStore("books");
```

### INDEXDB. Транзакции

это группа операций, это группа операций, которые должны быть выполнены все
db.transaction(store[, type]);

- **store** – это название хранилища, к которому транзакция получит доступ, например, "books". Может быть массивом названий, если нам нужно предоставить доступ к нескольким хранилищам.
- **type** – тип транзакции, один из:
- **readonly** – только чтение, по умолчанию.
- **readwrite** – только чтение и запись данных, создание/удаление самих хранилищ объектов недоступно.

```js
let transaction = db.transaction("books", "readwrite"); // Создать транзакцию и указать все хранилища, к которым необходим доступ, строка
// получить хранилище объектов для работы с ним
let books = transaction.objectStore("books"); //Получить хранилище объектов, используя transaction.objectStore(name)
let book = {
  id: "js",
  price: 10,
  created: new Date(),
};
let request = books.add(book); //Выполнить запрос на добавление элемента в хранилище объектов books.add(book)
request.onsuccess = function () {
  //Обработать результат запроса (4), затем мы можем выполнить другие запросы и так далее.
  console.log("Книга добавлена в хранилище", request.result);
};
request.onerror = function () {
  console.log("Ошибка", request.error);
};
```
