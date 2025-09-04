# CORS

Пример атаки

- вход на bank.com
- переход на evil.com
- - evil.com отправляет запрос на /api/account на bank.com.
- - браузер отправляет куки

Цель - защитить от отправки данных на вредоносные сайты

принцип работы:

- браузер делает запрос
- если get запрос, то браузер сам решит дать ответ или нет
- - блокируют чтение результата
- если post и другие запросы, сначала улетает options
- - получает Access-Control
- - проверяет разрешенные заголовки

# заголовки

```bash
# Разрешённые домены (один, список или *)
Access-Control-Allow-Origin: https://frontend.com

# Разрешённые HTTP-методы (список или *)
Access-Control-Allow-Methods: POST, GET, OPTIONS, DELETE

# Разрешённые заголовки для отправки(предварительный запрос)
Access-Control-Allow-Headers: Authorization, Content-Type, X-Requested-With

# Разрешённые для чтения заголовки (основной запрос)
Access-Control-Expose-Headers: Authorization, Content-Type, X-Requested-With

# Разрешить передачу кук/токенов
Access-Control-Allow-Credentials: true

# Кэшировать предварительный запрос на 600 сек (10 мин)
Access-Control-Max-Age: 600
```

# источники

одинаковые:

- https://site.com/page и https://site.com/about (отличаются путем или аргументами)

разные:

- http://site.com и https://site.com (разный протокол)
- https://site.com и https://api.site.com (разный домен)
