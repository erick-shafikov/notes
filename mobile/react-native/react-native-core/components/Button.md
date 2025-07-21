# Button

```jsx
<Button
  onPress={({nativeEvent: PressEvent})=>void} //(required) коллбек на нажатие
  title={'string'} //(required) текст
  color={"hexColor"} //цвет кнопки '#2196F3'  для Android и '#007AFF' IOS
  disabled={true} //активность кнопки
  touchSoundDisabled={false} //(Android) при нажатии воспроизводить звук
/>
```
