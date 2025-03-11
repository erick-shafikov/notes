## fetch – встроенная функция для загрузки данных.

```js
fetch(url, {
  cache: "auto no cache", //"force-cache" | "no-store",
  next: {
    revalidate: false, //false === revalidate: Infinity | 0 | number,
    tags: ["collection"],
  },
});
```
