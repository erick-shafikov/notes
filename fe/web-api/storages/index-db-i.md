# INDEXDB

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

## INDEXDB. Проблема параллельного обновления

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

## INDEXDB. Хранилище объектов

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

## INDEXDB. Транзакции

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
