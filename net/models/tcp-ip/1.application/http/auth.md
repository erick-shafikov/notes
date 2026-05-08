# аутентификация

сценарии как сервер распознает браузер, контролируется такими заголовками как WWW-Authenticate, Proxy-Authenticate. Существую разные варианты аутентификации: Basic, Bearer, Digest, HOBA, Mutual, AWS4-HMAC-SHA256

# Общая структура HTTP-аутентификации

Структура аутентификации между клиентом и сервером:

- Сервер отвечает 401 с помощью заголовка [WWW-Authenticate](./headers/res-headers.md#www-authenticate)
- Клиент отправляет [Authorization](./headers/req-headers.md#authorization)
- клиент запрашивает у пользователя пароль, а затем отправляет запрос, включающий корректный заголовок Authorization.

Такая же схема используется в прокси-аутентификации где используются [proxy-authenticate](./headers/res-headers.md#proxy-authenticate) и [proxy-authorization](./headers/req-headers.md#proxy-authorization) заголовки
