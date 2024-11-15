## useActionState

это хук, который позволяет обновлять состояние на основе результата действия формы.

```js
const [state, formAction] = useActionState(fn, initialState, permalink?);
//fn - функция при вызове отправки
//initialState - начальное состояние формы
// state - текущее состояние
// formAction - действие
// permalink - ссылка для изменения формы
```

Применение отображать результат отправки формы

```jsx
import { useActionState } from "react";

async function increment(previousState, formData) {
  return previousState + 1;
}

function StatefulForm({}) {
  const [state, formAction] = useActionState(increment, 0);
  return (
    <form>
      {state}
      <button formAction={formAction}>Increment</button>
    </form>
  );
}
```
