# Service Worker

Прокси между сервером и браузером

Регистрация:

```js
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/service-worker.js")
    .then(function (registration) {
      console.log(
        "Service Worker registration successful with scope: ",
        registration.scope
      );
    })
    .catch(function (error) {
      console.log("Service Worker registration failed: ", error);
    });
}
```

```js
self.addEventListener("fetch", function (event) {
  event.respondWith(function_that_returnedResponse());
});
```
