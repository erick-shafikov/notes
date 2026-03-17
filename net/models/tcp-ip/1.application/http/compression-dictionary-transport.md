# Compression Dictionary Transport (-ff, -sf)

призван уменьшить размер передаваемых сообщение, пример

```js
function a() {
  console.log("Hello World!");
}

function b() {
  console.log("I am here");
}
```

```bash
function a() {
  console.log("Hello World!");
}

[0:9]b[10:20]I am here[42:46]
```

в отличие от Brotli compression и Zstandard compression позволяет не копировать
