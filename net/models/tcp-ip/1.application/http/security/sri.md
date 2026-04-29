# Subresource Integrity

Subresource Integrity (SRI) — это функция безопасности, которая позволяет браузерам проверять, что ресурсы, которые они загружают (например, из CDN), доставляются без непредвиденных манипуляций. Принцип работы заключается в предоставлении криптографического хеша, которому должен соответствовать полученный ресурс.

Использование с элементами script, link у которых значение атрибута rel равно stylesheet, preload или modulepreload.

```html
<script
  src="https://cdn.example.com/script.js"
  integrity="
  sha384-Tk2Yjg3YmYzMWNkZTdhMTFkM2FlNDg4ZjE3MzEzNTk3ZDlh
  sha384-DEzZmZhMGFkMGQ0OTQ3MzZkNGY0OTg4NGIwN2ZiMMTM3YmQ
  sha512-ZmQ5NjNiYWJjYTM3MjRhMGI4MTQzNWRmZTZkZGYyMzQyOGYYTZkYjBm
  sha512-OGUwYThkZDc2YzFlZGI5MDEzZmZhMGFkMGQ0OTQ3MzZkNGYZTEzODk2"
  crossorigin="anonymous"
></script>
```

Cross-origin requests that use subresource integrity must use the Cross-Origin Resource Sharing (CORS) protocol.Это достигается путем отправки соответствующего заголовка ответа Access-Control-Allow-Origin.

Заголовки [Integrity-Policy](../headers/res-headers.md#integrity-policy) and [Integrity-Policy-Report-Only](!!!TODO link) сигнализируют о том что SRI - обязательна для ресурсов
