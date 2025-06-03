# bodyParser

```ts
const bodyParser = require("body-parser"); //подключение к библиотеки
app.use(
  bodyParser.urlencoded({
    //подключение к приложению
    extended: true,
  })
);
app.post("/", function (req, res) {
  const itemName = req.body.newItem; //использование
  // …
});
```

```html
<!-- в html форме: -->
<form class="item" action="/" method="post">
  <input type="text" name="newItem" placeholder="New Item" autocomplete="off" />
  <button type="submit" name="list" value="<%= listTitle %>">+</button>
</form>
```

## Express. dirname

автоматически определяет путь вне зависимости от инстанции, на которой запущен сервер

```js
const express = require("express");
const bodyParser = require("body-parser");
const request = require("request");
const app = express();
app.use(express.static("public"));
app.get("/", (req, res) => {
  res.sendFile(__dirname + "/signup.html"); //отправит в качестве запроса
});
app.listen("3000", () => {
  console.log("OK");
});
```

# EXPRESS. static

/project

- ↳/public
- ↳/css
- ↳style.css
- index.html :

```html
<link href="css/style.css" rel="stylesheet" />
```

```js
const express = require("express");
const bodyParser = require("body-parser");
const request = require("request");
const app = express();
app.use(express.static("public")); //в папке с проектом должна быть папка public. Путь во всех файлах, которые ссылаются на корневой каталог, ссылаются на public - директорию
app.get("/", (req, res) => {
  res.sendFile(__dirname + "/signup.html");
});
app.listen("3000", () => {
  console.log("OK");
});
```

# EXPRESS. req.params

```js
app.get("/:post", (req, res) => {
  console.log(req.params.post); //позволяет достать параметры url на лету
});
```

# typescript и express

```ts
import express, { Request, Response, NextFunction } from "express"; //экспортируем типы
import { userRouter } from "./users/users.js";

// const port = 8000;
// const app = express();
// app.use((req, res, next) => {
//   //пример middleware
//   console.log("Время", Date.now());
//   next();
// });

app.get("/hello", (req, res, next) => {
  //get в свою очередь типизирован
  throw new Error("Error");
});

// app.use("/users", userRouter); //кода пользователь переходит на вкладку user то мы перенаправляем его

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  //пример отлавливания ошибок
  console.log(err.message);
  res.status(401).send(err.message);
});
// app.listen(port, () => {
//   console.log(`Сервер запущен на http://localhost800:${port}`);
// });
```

# EJS

npm i ejs – установка пакета, создать папку view

```js
app.set('view engine', 'ejs') – подключение ejs модуля к приложению app
res.render("list", { kindOfВay: day }); // метод рендера по темплейту list, где в логике приложения расчитывается переменная day,

<h1>It's a <%=kindOfВay%>!</h1> // место для ejs модуля а file.ejs файле

<%= variable %> // для переменных
<% } else { %> // для каждой строчки кода, нужно открывать на каждой строке кода JS
```
