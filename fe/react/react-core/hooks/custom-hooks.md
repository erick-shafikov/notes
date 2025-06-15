## useDifferedValue

```tsx
const useDifferedValue = <T extends string | number>(value: T) => {
  const [returnValue, setReturnValue] = useState(value);
  const [ignore, setIgnore] = useState(true);

  useEffect(() => {
    const id = setTimeout(() => {
      setIgnore(false);
    }, 2000);
    return () => clearInterval(id);
  }, [value]);

  useEffect(() => {
    if (!ignore) {
      setReturnValue(value);
    }
    return () => {
      setIgnore(true);
    };
  }, [ignore]);

  return returnValue;
};
```

## Хук для fetch данных

```js
function useFetchData({ country }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let ignore = false;

    fetch(`/api/cities?country=${country}`)
      .then((response) => response.json())
      .then((json) => {
        if (!ignore) {
          setData(json);
        }
      });
    return () => {
      //для dev режима
      ignore = true;
    };
  }, [country]);
}
```

Такой подход имеют проблему водопада подгружаемых данных, если есть вложенные запросы. Решение:
Suspense (обернуть один в другой)
Использовать promise.all

## useDebounce

```jsx
const useDebounce = (callback, interval = 0) => {
  const prevTimeoutIdRef = React.useRef();

  return React.useCallback(
    (...args) => {
      clearTimeout(prevTimeoutIdRef.current);
      prevTimeoutIdRef.current = setTimeout(() => {
        clearTimeout(prevTimeoutIdRef.current);
        callback(...args);
      }, interval);
    },
    [callback, interval]
  );
};
```
