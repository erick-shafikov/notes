<!-- matchMedia ---------------------------------------------------------------------------------------------------------------------------------->

# matchMedia

```js
// для работы с медиа выражениям в js
var mediaQueryList = window.matchMedia("(orientation: portrait)");

if (mediaQueryList.matches) {
  /* Окно просмотра в настоящее время находится в книжной ориентации */
} else {
  /* Окно просмотра в настоящее время находится в альбомной ориентации */
}

var mediaQueryList = window.matchMedia("(orientation: portrait)"); // Создание списка выражений.
function handleOrientationChange(evt) {
  // Определение колбэк-функции для обработчика событий.
  if (evt.matches) {
    /* Окно просмотра в настоящее время находится в книжной ориентации */
  } else {
    /* Окно просмотра в настоящее время находится в альбомной ориентации */
  }
}

mediaQueryList.addListener(handleOrientationChange); // Добавление колбэк-функции в качестве обработчика к списку выражений.

handleOrientationChange(mediaQueryList); // Запуск обработчика изменений, один раз.
mediaQueryList.removeListener(handleOrientationChange);
```
