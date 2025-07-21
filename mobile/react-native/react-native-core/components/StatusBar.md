# StatusBar

Компонент статус бара

```js
<StatusBar
  animated={false}
  barStyle={"default" | "light-content" | "dark-content"} //стиль
  hidden={false}
  //Android only props
  currentHeight={Number}
  backgroundColor={"black"}
  translucent={false}
  //IOS only props
  networkActivityIndicatorVisible={false} //отображение индикатора сети
  showHideTransition={"fade" | "slide" | "none"} //анимация скрытия
/>
```

Методы

```js
popStackEntry(entry: StatusBarProps); //удаление стека
pushStackEntry();
replaceStackEntry();
setBackgroundColor();
setBarStyle();
setHidden();
setNetworkActivityIndicatorVisible() //(iOS)
setTranslucent(); //(Android)
```
