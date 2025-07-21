## useWindowDimensions

- useWindowDimensions : ScaledSize Используется для измерений, которые нужно каждый рендер в отличает от Dimensions

```js
interface ScaledSize {
  width: number;
  height: number;
  scale: number;
  fontScale: number;
}
//в компоненте
const { width, height } = useWindowDimensions();
//контрольные измерения
const marginTopDistance = height < 380 ? 30 : 100;
//применение
<View style={[styles.rootContainer, { marginTop: marginTopDistance }]}>
```
