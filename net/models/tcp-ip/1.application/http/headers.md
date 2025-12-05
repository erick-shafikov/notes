# подсказки

Заголовки по типам:

- Заголовки устройства (отправляет браузер, далее б):
- - Sec-CH-UA-\* - подсказки клиента User Agent
- - Sec-CH-UA - версия браузера
- - Sec-CH-UA-Platform - платформа
- - Sec-CH-UA-Mobile: ?1 или ?2 - работает ли на мобильном устройстве
- - Sec-CH-UA-Model - модель
- - Sec-CH-UA-Form-Factors - форм факторы
- запрос на доп заголовки (от сервера, далее с)
- - Accept-CH: Sec-CH-UA-Model, Sec-CH-UA-Form-Factors, DPR, Viewport-Width, Width - сервер требует доп информацию, браузер на каждый запрос будет отправлять Sec-CH-UA-Model, Sec-CH-UA-Form-Factors. Поведение можно реализовать в html:
    ```html
    <meta http-equiv="Accept-CH" content="Width, Downlink, Sec-CH-UA" />
    ```
- Кеширование:
- - Vary
- Критические
- - Sec-CH-Prefers-Reduced-Motion - может использоваться в Accept-CH
- Заголовки согласования контента - не являются частью стандарта, сервер сам определяет какой контент отдать:
- - Accept - mime типы
- - Accept-Encoding
- - Accept-Language
- - User-Agent
- - Vary
- Апгрейд протокола
- - Connection: upgrade - вернет 101 статус если изменит протокол или , также есть перечень заголовков для websocket протокола
- - Upgrade: example/1, foo/2
