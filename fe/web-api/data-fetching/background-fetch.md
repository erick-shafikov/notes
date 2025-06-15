# background fetch (-ff, -s)

- позволяет делать долгие запросы в фоновом режиме, загрузка будет происходит по мере подключения к сети

# BackgroundFetchManager

коллекция фоновых запросов

методы:

- - fetch(id, requests, options) ⇒ промис с результатом BackgroundFetchRegistration
- - get() ⇒ BackgroundFetchRegistration
- - getIDs() ⇒ запросов

# BackgroundFetchRegistration

фоновый запрос что возвращает fetch и get

свойства:

- id
- uploadTotal - кол-во байт
- uploaded - кол-во отпарвленных
- downloadTotal - общий размер загрузки
- downloaded - кол-во скаченных
- result - "success", либо "failure"
- failureReason
- recordsAvailable

методы:

- abort() - прерывает
- match() ⇒ BackgroundFetchRecord который подходит по аргументам
- matchAll() ⇒ массив BackgroundFetchRecord

события:

- progress - при изменение свойств uploaded, downloaded, result, failureReason

# BackgroundFetchRecord

запрос и ответ создается BackgroundFetchManager.fetch()

свойства:

- request
- responseReady ⇒ промис с результатом Response.

```js
bgFetch.match("/ep-5.mp3").then(async (record) => {
  if (!record) {
    console.log("Запись не найдена");
    return;
  }

  console.log(`Запрос`, record.request);

  const response = await record.responseReady;
  console.log(`Ответ`, response);
});
```

# BackgroundFetchEvent

событие (onbackgroundfetchabort, onbackgroundfetchclick)

одно свойство registration ⇒ BackgroundFetchRegistration для которого вызван

# BackgroundFetchUpdateUIEvent

событие, которое передается в onbackgroundfetchsuccess и onbackgroundfetchfail.

метод:

- updateUI() - покажет иконку загрузки
