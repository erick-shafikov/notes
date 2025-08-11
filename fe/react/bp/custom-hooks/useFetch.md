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
