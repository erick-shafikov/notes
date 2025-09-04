# useReducer

Позволяет вынести логику из UI компонента

```js
const [state, dispatch] = useReducer(reducer, initialArg, init?)
//reducer - функция, которая показывает как будет меняться состояние
//initialArg - изначальное состояние (если инициировать функцией, то будет вызывать перерендер)
//init - функция которая позволяет инициализировать состояние initialArg попадет как аргумент

//state - состояние
//dispatch - функция, которая изменяет состояние
```

- При инициализации вызывается два раза в строгом режиме
- нельзя мутировать состояние

```jsx
import { useReducer } from "react";

function reducer(state, action) {
  switch (action.type) {
    case "incremented_age": {
      return {
        name: state.name,
        age: state.age + 1,
      };
    }
    case "changed_name": {
      return {
        name: action.nextName,
        age: state.age,
      };
    }
  }
  throw Error("Unknown action: " + action.type);
}

const initialState = { name: "Taylor", age: 42 };

export default function Form() {
  const [state, dispatch] = useReducer(reducer, initialState);

  function handleButtonClick() {
    dispatch({ type: "incremented_age" });
  }

  function handleInputChange(e) {
    dispatch({
      type: "changed_name",
      nextName: e.target.value,
    });
  }

  return (
    <>
      <input value={state.name} onChange={handleInputChange} />
      <button onClick={handleButtonClick}>Increment age</button>
      <p>
        Hello, {state.name}. You are {state.age}.
      </p>
    </>
  );
}
```

вычисление следующего состояния в текущем

```js
const action = { type: "incremented_age" };
dispatch(action);

const nextState = reducer(state, action);
console.log(state); // { age: 42 }
console.log(nextState); // { age: 43 }
```

# начальное состояние

Возникающие проблемы:

- Мутация объекта: Если initialFormState изменится где-то в приложении, вы больше не получите “чистое” начальное состояние при его повторном применении;
- Ошибки при сбросе: Попытка сбросить форму к начальному состоянию приведёт к тому, что вы получите уже изменённый объект вместо оригинального;
- Непредсказуемость в тестах: Множество юнит-тестов могут модифицировать это состояние, что приведёт к поломанным тестам при одновременном запуске.

Почему это плохо

- Потеря чистоты: Вы не контролируете, что именно попадёт в начальное состояние при каждом новом его получении;
- Скрытые баги: Возникают “плавающие” баги, которые сложно отследить — особенно в больших командах и при автоматизированных тестах.

```js
const getInitialFormState = () => ({
  text: "",
  error: "",
  touched: false,
});

const formReducer = (state, action) => {
  // Логика редьюсера
};

const Component = () => {
  const [state, dispatch] = useReducer(formReducer, getInitialFormState());
  // Остальной код
};
```
