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

# throttle

```js
function throttle(func, limit) {
  let inThrottle;

  return function (...args) {
    if (!inThrottle) {
      func.apply(this, args);
      let throttle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

const throttled = debounce(() => console.log("x"), 1000);
throttled();
```
