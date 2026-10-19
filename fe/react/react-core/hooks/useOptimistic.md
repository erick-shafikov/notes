## useOptimistic

позволяет показывать некоторое состояния во время асинхронного действия

```tsx
const [optimisticState, addOptimistic] = useOptimistic(state, updateFn);
//state - изначальное состояние
//updateFn(currentState, optimisticValue) - принимает два аргумента, которые при объединение дадут возвращаемое значение
//optimisticState - состояние на отображение
// addOptimistic - функция, которая добавляет новое состояние
```

```tsx
import { useOptimistic, useState, useRef } from "react";
import { deliverMessage } from "./actions.js";

interface Message {
  text: string;
  sending?: boolean;
}

interface ThreadProps {
  messages: Message[];
  sendMessage: (formData: FormData) => Promise<void>;
}

//messages - изначальное отображение 1.1[2]
function Thread({ messages, sendMessage }: ThreadProps): JSX.Element {
  const formRef = useRef<HTMLFormElement>(null);

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

export default function App(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([
    { text: "Hello there!", sending: false },
  ]);
  async function sendMessage(formData: FormData): Promise<void> {
    const sentMessage = await deliverMessage(formData.get("message"));
    setMessages((messages) => [...messages, { text: sentMessage }]);
  }
  return <Thread messages={messages} sendMessage={sendMessage} />;
}
```

```tsx
const [optimisticState, addOptimistic] = useOptimistic(state, updateFn);
```

с асинхронными функциями

```tsx
interface ChangeNameProps {
  currentName: string;
  onUpdateName: (name: string) => void;
}

function ChangeName({ currentName, onUpdateName }: ChangeNameProps): JSX.Element {
  const [optimisticName, setOptimisticName] = useOptimistic(currentName);

  const submitAction = async (formData: FormData): Promise<void> => {
    //получение нового значения
    const newName = formData.get("name") as string;
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
