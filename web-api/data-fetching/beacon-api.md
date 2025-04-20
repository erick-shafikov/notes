неблокирующие POST - запросы, использующиеся для отправки статистики итд

Navigator.sendBeacon(url, data) - принимает два параметра - URL и данные для отправки в запросе

```js
window.addEventListener("unload", logData, false);

function logData() {
  navigator.sendBeacon("/log", analyticsData);
}
```
