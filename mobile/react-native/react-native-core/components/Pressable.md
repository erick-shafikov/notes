# Pressable

Обертка для объектов для того, что бы их сделать кликабельными

```js
<Pressable
  android_ripple={{//(Android) стилизация при нажатии на андроиде
    color:'color',
    borderless: 'boolean'
    radius: 'number'
    foreground: 'boolean'
  }}
  android_disableSound={false}
  unstable_pressDelay={Number}// настройка для onLongPress
  delayLongPress={Number}
  disabled={false}
  hitSlop={{ // дистанция от элемента, при которой сработает
    bottom: 20,
    left: null,
    right: undefined,
    top: 50
}}
// для hover web версии
  onHoverIn={({ nativeEvent: MouseEvent }) => undefined}
  onHoverOut={({ nativeEvent: MouseEvent }) => undefined} // если держать палец больше 500 мс
  onLongPress={({nativeEvent: PressEvent}) => undefined} // onPressIn -> onPressOut
  onPress={({nativeEvent: PressEvent}) => undefined} //вызывается при нажатии
  onPressIn={({nativeEvent: PressEvent}) => undefined} //вызывается при отпускании
  onPressOut={({nativeEvent: PressEvent}) => undefined}// как далеко считать onPressOut
  pressRetentionOffset={{bottom: 30, left: 20, right: 20, top: 20,}}
  _style={({ pressed }) => ViewStyle} //стиль может принимать функцию, которая принимает параметр pressed
  // пример
  _style={({ pressed }) => [
    styles.button,
    pressed ? styles.buttonPressed : null,
  ]}
 >
```
