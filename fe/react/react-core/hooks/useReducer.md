# useReducer

Позволяет вынести логику из UI компонента

```js
const [state, dispatch] = useReducer(reducer, initialArg, init?)
//reducer - функция, которая показывает как будет меняться состояние
//initialArg - изначальное состояние (если инициировать функцией, то будет вызывать перерендер)
//init - функция которая позволяет инициализировать состояние

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
