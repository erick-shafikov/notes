## useOptimistic

позволяет показывать некоторое состояния во время асинхронного действия

```js
const [optimisticState, addOptimistic] = useOptimistic(state, updateFn);
//state - изначальное состояние
//updateFn(currentState, optimisticValue) - принимает два аргумента, которые при объединение дадут возвращаемое значение
//optimisticState - состояние на отображение
// addOptimistic - функция, которая добавляет новое состояние
```

```jsx
import { useOptimistic, useState, useRef } from "react";
import { deliverMessage } from "./actions.js";

//messages - изначальное отображение 1.1[2]
function Thread({ messages, sendMessage }) {
  const formRef = useRef();

  //функция для добавления асинхронного действия
  async function formAction(formData) {
    // 2.1[1] добавляет в состояние переданные данные в форму
    addOptimisticMessage(formData.get("message"));
    formRef.current.reset();
    await sendMessage(formData);
  }

  //optimisticMessages - итоговые значения
  // функция, которая добавляет оптимистичное значение 2.1[2]
  const [optimisticMessages, addOptimisticMessage] = useOptimistic(
    // изначальные значения 1.2[2]
    messages,
    // коллбек который изменяет состояние, принимает текущее и оптимистичное состояние
    (state, newMessage) => [
      ...state,
      {
        text: newMessage,
        sending: true,
      },
    ]
  );

  return (
    <>
      {optimisticMessages.map((message, index) => (
        <div key={index}>
          {message.text}
          {!!message.sending && <small> (Sending...)</small>}
        </div>
      ))}
      <form action={formAction} ref={formRef}>
        <input type="text" name="message" placeholder="Hello!" />
        <button type="submit">Send</button>
      </form>
    </>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { text: "Hello there!", sending: false, key: 1 },
  ]);
  async function sendMessage(formData) {
    const sentMessage = await deliverMessage(formData.get("message"));
    setMessages((messages) => [...messages, { text: sentMessage }]);
  }
  return <Thread messages={messages} sendMessage={sendMessage} />;
}
```

```js
const [optimisticState, addOptimistic] = useOptimistic(state, updateFn);
```

с асинхронными функциями

```js
function ChangeName({ currentName, onUpdateName }) {
  const [optimisticName, setOptimisticName] = useOptimistic(currentName);

  const submitAction = async (formData) => {
    //получение нового значения
    const newName = formData.get("name");
    //использование значения
    setOptimisticName(newName);
    const updatedName = await updateName(newName);
    onUpdateName(updatedName);
  };

  return (
    <form action={submitAction}>
      <p>Your name is: {optimisticName}</p>
      <p>
        <label>Change Name:</label>
        <input
          type="text"
          name="name"
          disabled={currentName !== optimisticName}
        />
      </p>
    </form>
  );
}
```
