# Events

Что бы инициировать событие

```js
type T1 = fireEvent(node: HTMLElement, event: Event)

// второй вариант вызова
type T2 = fireEvent[eventName](node: HTMLElement, eventProperties: Object)

// <button>Submit</button>
fireEvent(
  getByText(container, 'Submit'),
  new MouseEvent('click', {
    bubbles: true,
    cancelable: true,
  }),
)
```

ограничения:

- Keydown

```js
// -
fireEvent.keyDown(getByText("click me"));
// +
getByText("click me").focus();
fireEvent.keyDown(document.activeElement || document.body);
```

- Focus/Blur

```js
// -
fireEvent.focus(getByText("focus me"));
// +
getByText("focus me").focus();
```
