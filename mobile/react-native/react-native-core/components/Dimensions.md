# Dimensions

Dimensions - объект, который возвращает параметры экрана. Используется при определении размеров, которые требуются при первоначальной инициализации компонента

```js
//узнаем ширину экрана возвращает { width: number; height: number; scale: number; fontScale: number; }
const deviceWidth = Dimensions.get("window").width;
//использование
height: deviceWidth < 380 ? 160 : 300,
```

методы

```js
addEventListener(type: 'change', handler: ({window, screen} : DimensionValue) => void) //при изменении
get(dim: 'window' | 'screen') : { //метод на получение информации
  width: Number,
  height: Number,
  scale: Number,
  fontScale: Number
}
```
