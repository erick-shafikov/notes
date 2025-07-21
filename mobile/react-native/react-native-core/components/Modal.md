# Modal

```js
<Modal
  animationType={"none" | "slide" | "fade"} // slide - выезжает снизу, fade -гаснет
  hardwareAccelerated={false} //(Android) аппаратное ускорение
  onDismiss={Function} //(IOS) функция на закрытие
  onOrientationChange={Function} //(IOS) сработает при смене ориентации
  onRequestClose={Function} //сработает при нажатии кнопки назад
  onShow={Function} //функция которая будет вызвана один раз при открытие
  presentationStyle={
    "fullScreen" | "pageSheet" | "formSheet" | "overFullScreen"
  } //(IOS)
  statusBarTranslucent={false} //(Android) должно ли диалоговое окно уходить
  supportedOrientations={
    "portrait" |
    "portrait-upside-down" |
    "landscape" |
    "landscape-left" |
    "landscape-right"
  } //(IOS)
  transparent={false} //прозрачный фон если true
  visible={true} //флаг отвечающий за видимость
/>
```

```js
const ModalButton = () => {
  const [modalVisible, setModalVisible] = useState(false);
  return (
    <View>
      //контейнер модального окна
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => {
          Alert.alert("Modal has been closed.");
          setModalVisible(!modalVisible);
        }}
      >
        // содержимое модального окна
        <View>
          <Text>Hello World!</Text>
          // Кнопка для закрытия
          <Pressable onPress={() => setModalVisible(!modalVisible)}>
            <Text>Hide Modal</Text>
          </Pressable>
        </View>
      </Modal>
      // Кнопка для открытия
      <Pressable onPress={() => setModalVisible(true)}>
        <Text>Show Modal</Text>
      </Pressable>
    </View>
  );
};
```
