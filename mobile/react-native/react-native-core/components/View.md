# View

основной контейнер - элемент
берет столько сколько можно

```js
<View

  accessible={true}//доступны ли для касания
  hitSlop={{ top: 10, bottom: 10, left: 0, right: 0 }}//насколько далеко от элемента будут срабатывать события касания
  onLayout={({nativeEvent: LayoutEvent}) => void}// вызовется, когда будут определены размеры

  // при движении пальца
  onMoveShouldSetResponder={({nativeEvent: PressEvent}) => boolean}// срабатывает на каждое касание
  onMoveShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}// если нужно предотвращение при движении от родителя, функция должна возвращать true
  onResponderGrant={({nativeEvent: PressEvent}) => void ｜ boolean}
  onResponderMove={({nativeEvent: PressEvent}) => void}
  onResponderReject={({nativeEvent: PressEvent}) => void}
  onResponderRelease={({nativeEvent: PressEvent}) => void}
  onResponderTerminate={({nativeEvent: PressEvent}) => void}
  onResponderTerminationRequest={({nativeEvent: PressEvent}) => void}
  onStartShouldSetResponder={({nativeEvent: PressEvent}) => boolean}
  onStartShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}
  //
  pointerEvents={'box-none'|'none'|'box-only'|'auto'}

  //Android only props
  tabIndex={0 | -1}
  //Style
  style={type ViewStyle}
/>
```
