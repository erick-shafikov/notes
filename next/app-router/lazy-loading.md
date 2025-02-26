## lazy loading

```js
//объявление в качестве переменной
const LazyComponent = dynamic(() => import("../components/Lazy/Lazy"), {
  ssr: false,
  loading: () => <>Loading...</>,
});
```
