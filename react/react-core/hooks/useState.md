# useState

Асинхронные хук
нельзя вызывать в логических блоках

```js
function Example() {
  // Объявляем новую переменную состояния "count"
  const [count, setCount] = useState(0); // state: {count:0}
  return (
    <div>
      <p>Вы нажали {count} раз</p>
      <button onClick={() => setCount(count + 1)}>Нажми на меня</button>
    </div>
  );
}
```

useState возвращает массив с двумя элементами – текущее значение состояния и функцию для его изменения
единственный аргумент – изначальное значение состояния

- объявлять setState внутри условных блоков нельзя
- если дебажить состояние после изменения, то оно будет предыдущее

```js
function ExampleWithManyStates() {
  // Объявляем несколько переменных состояния!
  const [age, setAge] = useState(42);
  const [fruit, setFruit] = useState("банан");
  const [todos, setTodos] = useState([{ text: "Изучить хуки" }]); // ...
}
```

## prevValue

Форма, которая динамически отображает изменения в двух полях

```jsx
function App() {
  let [fullName, setFullName] = useState({
    //исходное состояние
    fName: "",
    lName: "",
  });
  const changeHandler = (e) => {
    const { value, name } = e.target; //деструктуризация объекта e.target, так как одинаковые обработчики события (changeHandler) относятся к двум полям формы, то target.name будет определять какое поле изменяется

    setFullName((prevValue) => {
      //prevValue – это аргумент функции, который характеризует собой предыдущее состояние, ниже возвращаются в зависимости от ситуации два варианта объекта состояния
      if (name === "fName") {
        return {
          fName: value, //переписываем
          lName: prevValue.lName, //сохраняем старое значение
        };
      } else if (name === "lName") {
        return {
          fName: prevValue.fName,
          lName: value,
        };
      }
    });
  };
}
```

альтернатива двум if

```jsx
function App() {
  let [fullName, setFullName] = useState({
    //исходное состояние
    fName: "",
    lName: "",
  });
  const changeHandler = (e) => {
    const { value, name } = e.target; //деструктуризация объекта e.target, так как одинаковые обработчики события (changeHandler) относятся к двум полям формы, то target.name будет определять какое поле изменяется
    setFullName((prevValue) => {
      return {
        ...prevValue,
        [name]: value,
      };
    });
    //или
    setContact((prevValue) => ({ ...prevValue, [name]: value }));
  };
}
```

Создание предыдущего состояния

```js
// не ок
const [todos, setTodos] = useState(createInitialTodos());
// ок
const [todos, setTodos] = useState(createInitialTodos);
```

## BP

```js
import React, { useState } from "react";
import ToDoItem from "./ToDoItem";
function App() {
  const [inputText, setInputText] = useState("");
  const [items, setItems] = useState([]);
  function handleChange(event) {
    const newValue = event.target.value;
    setInputText(newValue);
  }
  function addItem() {
    setItems((prevItems) => {
      return [...prevItems, inputText];
    });
    setInputText("");
  }
  function deleteItem(id) {
    setItems((prevItems) => {
      return prevItems.filter((item, index) => {
        return index !== id;
      });
    });
  }
  return (
    <div className="container">
      <div className="heading">
        <h1>To-Do List</h1>
      </div>
      <div className="form">
        <input onChange={handleChange} type="text" value={inputText} /> 
        <button onClick={addItem}>
          <span>Add</span>
        </button>
      </div>
      <div>
        <ul>
          {items.map((todoItem, index) => (
            <ToDoItem
              text={todoItem}
              id={index}
              key={index}
              unChecked={deleteItem}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}
export default App;
```
