# Куки

При входе на сайт сервер отсылает в ответе HTTP-заголовок Set-Cookie, устанавливая специальный идентификатор сессии session identifier. Следующий запрос к тому же домену посылает заголовок HTTP Cookie

**чтение** – document.cookie

**Запись**
Запись обновляет только упомянутые куки, но не затрагивает все остальные
document.cookie = “user-John”;
имя и значения – любые символы – для формирования можно использовать encodeURIComponent. Ограничение – 4кб на одну пару, ограничение 20+ пар

Сервер отправляет:

```
HTTP/1.0 200 OK
Content-type: text/html
Set-Cookie: yummy_cookie=choco
Set-Cookie: tasty_cookie=strawberry
```

И при каждом запросе сервер будет получать

```
GET /sample_page.html HTTP/1.1
Host: www.example.org
Cookie: yummy_cookie=choco; tasty_cookie=strawberry
```

На стороне пользователя можно устанавливать c помощью js

```js
let name = "my name"; // специальные символы (пробелы), требуется кодирование
let value = "John Smith"; // кодирует в my%20name=John%20Smith
document.cookie = encodeURIComponent(name) + "=" + encodeURIComponent(value);
alert(document.cookie); // ...; my%20name=John%20Smith
```

## Настройки куки

```js
document.cookie = "user=John; path=/; expires=Tue, 19 Jan 2038 03:14:07 GMT";
```

### domain

domain=site.com – куки принадлежать только домену, который его установил. Нельзя сделать доступным куки на другом сайте

```js
document.cookie = "user=John"; // на site.com
alert(document.cookie); // нет user
```

если информацию нужно передать поддоменам

```js
// находясь на странице site.com сделаем куки доступным для всех поддоменов \*.site.com:
document.cookie = "user=John; domain=site.com";
alert(document.cookie); // позже на forum.site.com есть куки user=John
```

### expires, max-age

если нет ни одного из этих параметров, то куки удалятся после закрытия браузера для установки таймера удаления:

```js
expires=Tue, 19 Jan 2038 03:14:07 GMT
let date = new Date(Date.now() + 86400e3); // +1 день от текущей даты
date = date.toUTCString();
document.cookie = "user=John; expires=" + date;
```

### httpOnly

отключает JS от манипуляций с cookie, то есть document.cookie не видит эти cookie

### max-age

```js
document.cookie = "user=John; max-age=3600"; //куки будут удалены через час
document.cookie = "User=John; max-age=0"; //куки будут удалены сразу
```

Если эти параметры не указаны, то max-age === session

### path

url префикс пути, должен быть абсолютным
path=/admin то оно будет доступно на /admin и /admin/something, но не на страницах /home /page

### secure

Куки следует передавать только по HTTPS–протоколу, не будет доступно тому же сайту по протоколу http,
Secure флаг позволяет передавать куку только по HTTPS

```
Set-Cookie: id=a3fWa; Expires=Wed, 21 Oct 2015 07:28:00 GMT; Secure; HttpOnly
```

### samesite

Защита от XSRF атаки. Может принимать значения:

- Strict: (то же самое, что и samesite без значения) куки никуда не отправятся, если пользователь не пришел с того же сайта
- Lax: использует только безопасные методы GET но не POST
- None: отключает ограничение на отправку кук для межсайтовых запросов, но только в безопасном контексте (то есть если установлен SameSite=None, тогда также должен быть установлен атрибут Secure). Если атрибут SameSite не установлен, куки будут восприниматься как Lax

```
Set-Cookie: mykey=myvalue; SameSite=Strict
```

## Кастомные функции работы с куки

### getCookie(name)

```js
//возвращает cookie с заданным name
function getCookie(name) {
  let matches = document.cookie.match(
    new RegExp(
      "(?:^|; )" +
        name.replace(/([\.$?*|{}\(\)\[\]\\\/\+^])/g, "\\$1") +
        "=([^;]*)" //генерация name=<value>
    )
  );
  return matches ? decodeURIComponent(matches[1]) : undefined;
}
```

## setCookie(name, options, options={})

```js
//  устанавливает куки с именем value с настройкой path=/
function setCookie(name, value, options = {}) {
  options = {
    path: "/", // при необходимости добавьте другие значения по умолчанию
    ...options,
  };
  if (options.expires instanceof Date) {
    options.expires = options.expires.toUTCString();
  }
  let updatedCookie =
    encodeURIComponent(name) + "=" + encodeURIComponent(value);
  for (let optionKey in options) {
    updatedCookie += "; " + optionKey;
    let optionValue = options[optionKey];
    if (optionValue !== true) {
      updatedCookie += "=" + optionValue;
    }
  }
  document.cookie = updatedCookie;
}
setCookie("user", "John", { secure: true, "max-age": 3600 }); // Пример использования:
```

## deleteCookie(name)

```js
//Чтобы удалить куки, можно установить отрицательную дату истечения срока
function deleteCookie(name) {
  setCookie(name, "", {
    "max-age": -1,
  });
}
```
