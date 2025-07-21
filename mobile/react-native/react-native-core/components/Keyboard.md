# Keyboard

Работа с клавиатурой осуществляется с помощью событийной модели

```js
const Example = () => {
  const [keyboardStatus, setKeyboardStatus] = useState("");

  useEffect(() => {
    const showSubscription = Keyboard.addListener("keyboardDidShow", () => {
      setKeyboardStatus("Keyboard Shown");
    });
    const hideSubscription = Keyboard.addListener("keyboardDidHide", () => {
      setKeyboardStatus("Keyboard Hidden");
    });

    return () => {
      showSubscription.remove();
      hideSubscription.remove();
    };
  }, []);

  return (
    <View styles={style.container}>
      <TextInput
        styles={style.input}
        placeholder="Click here…"
        onSubmitEditing={Keyboard.dismiss}
      />
      <Text styles={style.status}>{keyboardStatus}</Text>
    </View>
  );
};
```

Методы

```js
addListener( eventType:
| 'keyboardWillShow'
| 'keyboardDidShow'
| 'keyboardWillHide'
| 'keyboardDidHide'
| 'keyboardWillChangeFrame'
| 'keyboardDidChangeFrame');
scheduleLayoutAnimation(event: KeyboardEvent)
dismiss();
isVisible();
metrics();
```
