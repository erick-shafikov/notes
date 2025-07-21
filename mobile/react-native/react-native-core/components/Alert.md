# Alert

Диалоговое окно с предупреждением. На ios можно любое количество кнопок. Для Android две кнопки OK и Cancel

```js
Alert.alert(
  title: String, //текст заголовка
  message: String, //сообщение
  buttons: type AlertButton[], //кнопки снизу от сообщения
  options: type AlertOption //конфигурация
);

Alert.prompt({
  title: String, //титул диалогового окна
  message: String, //Сообщение перед текстовым вводом
  callbackOptions: (text: String) => void | AlertButton[],
  type: 'default' | 'plain-text' | 'secure-text' | 'login-password' //IOS тип ввода
  defaultValue: String, //текст по умолчанию в поле ввода
  keyboardType: String //вариант раскладки клавиатуры
})

type AlertButtonStyle =  'default' | 'cancel' | 'destructive'

type AlertButton = {
  text: String;
  onPress: Function;
  style: 'default' | 'cancel' | 'destructive' //(IOS) вид кнопки
  isPreferred: false //(IOS) подчеркнутость кнопки
}

type AlertOption = {
  cancelable: false; //(Android) закрытие при клике
  userInterfaceStyle: 'light' | 'dark';
  onDismiss: Function //(Android) функция, которая срабатывает на закрытие
}
```
