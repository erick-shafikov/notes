# STORAGE. ХРАНЕНИЕ ДАННЫХ В БРАУЗЕРЕ

# local и session storage

отличие от cookie – не отправляют запрос на сервер
сервер не может манипулировать через HTTP-заголовки

setItem(key, value) – сохранить пару ключ/значение
getItem(key) – получить данные по ключу key
removeItem(key) – удалить данные с ключом key
clear() – удалить все
key(index) – получить ключ на заданной позиции
length – количество элементов в хранилище

## Демо localStorage

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

## Перебор ключей

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

# Только строки

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

# sessionStorage

существует только в рамках текущей вкладки, другая вкладка будет иметь другое хранилище
разделяется между фреймами, если они из одного и того же источника

```js
sessionStorage.setItem("test", 1);
alert(sessionStorage.getItem("test"));
```

но если открыть в другой вкладке – данные не будут доступны (null)

# Событие storage

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
