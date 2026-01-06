# debounce

```js
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);

    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

const debounced = debounce(() => console.log("x"), 1000);
debounced();
```
