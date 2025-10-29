# FETCH

Сетевые запросы – AJAX – Asynchronous JS & XML

Синтаксис: let promise = fetch(url, [options])
url – url для отправки запроса
без options – простой get запрос

- Возвращает промис
- Получения ответа два этапа – promise выполняется с объектом внутреннего класса Response в качестве результата, как только пришли заголовки ответа, можно проверить статус HTTP запроса

```js
let response = await fetch(url);
if (response.ok) {
  // если HTTP-статус в диапазоне 200-299
  let json = await response.json(); //получаем тело ответа (см. про этот метод ниже), также существуют методы: response.text() для обычного текста, response.formData(), response.blob(), response.arrayBuffer()
} else {
  alert("Ошибка HTTP: " + response.status);
}
```

```js
let url = "...";
let response = await fetch(url);
let commits = await response.json(); // читаем ответ в формате JSON
alert(commits[0].author.login);

// через цепочку промисов
fetch(
  "https://api.github.com/repos/javascript-tutorial/en.javascript.info/commits"
)
  .then((response) => response.json())
  .then((commits) => alert(commits[0].author.login));
```

# POST запрос

```js
let user = {
  name: "John",
  surname: "Smith",
};
let response = await fetch("...", {
  method: "POST",
  headers: {
    "Content-Type": "aplication/json; charset=utf-8",
  },
  body: JSON.stringify(user),
});

let result = await response.json();
alert(result.message);
```

# заголовки

Заголовки ответов хранятся в объекте похожем на Map

```js
let response = await fetch("..."); // получить один заголовок
alert(response.headers.get("Content-Type")); // application/json; charset=utf-8
for (let [key, value] of response.headers) {
  // перебрать все заголовки
  alert(`${key} = ${value}`);
}
// Заголовки запросов
let response = fetch(protectedUrl, {
  headers: {
    Authentication: "secret",
  },
});
```

# Файлы

при передачи фалов все тело должно быть представлено в виде FormData

```ts
const formData = new FormData();
formData.append("some_field", "some_value");
formData.append("avatar", fileInput.files[0]);

fetch("/api/upload", {
  method: "POST",
  body: formData, // тело одно, типа multipart/form-data
  // Заголовки не нужно указывать, подставит автоматически
  // headers: {
  //   "Content-Type": "multipart/form-data",
  // },
});
```
