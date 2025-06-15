# preinit, preload, prefetchDNS, preconnect

позволяется пред загрузить контент

```js
import { prefetchDNS, preconnect, preload, preinit } from "react-dom";
function MyComponent() {
  preinit("https://.../path/to/some/script.js", { as: "script" }); // загружает и выполняет этот скрипт с нетерпением
  preload("https://.../path/to/font.woff", { as: "font" }); // предварительно загружает этот шрифт
  preload("https://.../path/to/stylesheet.css", { as: "style" }); // предварительно загружает эту таблицу стилей
  prefetchDNS("https://..."); // когда вы можете ничего не запрашивать у этого хоста.
  preconnect("https://..."); // когда вы хотите что-то попросить, но не уверены, что именно.
}
```

результат в html

```html
<html>
  <head>
    <!-- ссылки/скрипты приоритетны по их полезности для ранней загрузки, а не по порядку вызова -->
    <link rel="prefetch-dns" href="https://..." />
    <link rel="preconnect" href="https://..." />
    <link rel="preload" as="font" href="https://.../path/to/font.woff" />
    <link rel="preload" as="style" href="https://.../path/to/stylesheet.css" />
    <script async="" src="https://.../path/to/some/script.js"></script>
  </head>
  <body>
    ...
  </body>
</html>
```
