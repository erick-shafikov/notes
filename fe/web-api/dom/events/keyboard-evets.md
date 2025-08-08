# События клавиатуры

# События keydown и keyup

события keyup и keydown происходят при нажатии и отпускании клавиши

свойство key позволяет получить символ
event.key – это непосредственно символ event.code – всегда будет тот же
при "z" event.key == z , event.code = KeyZ
при "Z" event.key = Z , event.code = keyZ

Для буквенных клавиш – KeyA KeyB… для числовых Digit0, Digit1 …
Событие для комбинации клавиш ctrl + z

```js
document.addEventListener("keydown", function (event) {
  if (event.code == "KeyZ" && (event.ctrlKey || event.metaKey)) {
    alert("отменить");
  }
});
```

# Действия по умолчанию

Действия по умолчанию

появление символа, удаления символа, прокрутка страницы, открытие диалогового окна «Сохранить», что бы их предотвратить event.preventDefault(), но такие комбинации как alt+f4 нельзя предотвратить

```html
<script>
   //проверка на вводимые символы
  function checkKey(key) {
    return(key >= "0" && key <= "9") || key  == "+" || key == "(" || key = ")" || key == "-"
  }
</script>

<input
  onkeydown="return checkPhoneKey(event.key)"
  placeholder="Введите номер
телефона"
/>
```

# Отследить одновременное нажатие

```js
function runOnKeys(func, ...codes) {
  let pressed = new Set(); //сет с нажатыми клавишами

  document.addEventLIstener("keydown", function (event) {
    pressed.add(event.code); //при каждом нажатии клавиши клавиша добавляется в сет

    for (let code of codes) {
      //пробегаем по rest-массиву с заданными калвишами
      if (!pressed.has(code)) {
        //если хоть одной нет, то выход
        return;
      }
    }
    pressed.clear(); //скидываем комбинацию нажатых, очищая сет

    func(); //запускаем функцию
  });
}
document.addEventListener("keyup", function (event) {
  pressed.delete(event.code); //при поднятии клавиши удаляет ее из нажатых
});

runOnKeys(() => alert("Hi"), "keyQ", "keyW");
```
