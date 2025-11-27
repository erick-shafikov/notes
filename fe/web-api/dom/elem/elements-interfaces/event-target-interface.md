# EventTarget

это интерфейс, реализуемый объектами, которые могут генерировать события и могут иметь подписчиков на эти события

# конструктор

```js
class MyEventTarget extends EventTarget {
  constructor(mySecret) {
    super();
    this._secret = mySecret;
  }

  get secret() {
    return this._secret;
  }
}

let myEventTarget = new MyEventTarget(5);
let value = myEventTarget.secret; // == 5
myEventTarget.addEventListener("foo", function (e) {
  this._secret = e.detail;
});

let event = new CustomEvent("foo", { detail: 7 });
myEventTarget.dispatchEvent(event);
let newValue = myEventTarget.secret; // == 7
```

# методы экземпляра

- addEventListener(type, listener, useCapture)
- - type - название события
- - listener - функция обработки
- - useCapture может быть объектом:
- - - capture = false будет ли отправлено ниже по dom дереву
- - - once = false сработает один раз
- - - passive = true слушатель события не вызовет preventDefault()
- - - signal = для AbortSignal отмены
- - useCapture может быть boolean будет ли отправлено ниже по dom дереву
- dispatchEvent() - запускает событие
- removeEventListener() - удалит прослушивание события
