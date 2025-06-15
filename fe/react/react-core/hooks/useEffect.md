# useEffect

Все эффекты запускаются в конце коммита react, после обновления экрана

useEffect запускается дважды и если есть возвращаемый коллбек, то он вызывается после первого вызова перед следующим рендером. Это сделано для режима разработки, чтобы показать, что есть отписка от событий или нет

Стоит придерживаться логики, что одно действие – один эффект

Значения бывают реактивные (пропсы и состояние) и нереактивные (глобальные переменные)

синтаксис: useEffect(()=>{функция для исполнения}, [значение аргумента для сброса применения эффекта])

Выполняет туже роль, что и ComponentDidMount, ComponentDidUpdate. React запускает эффекты после каждого обновления страницы, включая первое

```jsx
import React, { useState, useEffect } from "react";
function Example() {
  const [count, setCount] = useState(0); // По принципу componentDidMount и componentDidUpdate:
  useEffect(() => {
    // Обновляем заголовок документа, используя API браузера
    document.title = `Вы нажали ${count} раз`;
  });
  return (
    <div>
      <p>Вы нажали {count} раз</p>
      <button onClick={() => setCount(count + 1)}>Нажми на меня</button>
    </div>
  );
}
```

Чтобы сделать что-то после рендера или монтирования нужно вернуть соответствующую функцию

```jsx
import React, { useState, useEffect } from "react";
function FriendStatus(props) {
  const [isOnline, setIsOnline] = useState(null);
  useEffect(() => {
    function handleStatusChange(status) {
      setIsOnline(status.isOnline);
    }
    ChatAPI.subscribeToFriendStatus(props.friend.id, handleStatusChange); // Указываем, как сбросить этот эффект:
    return function cleanup() {
      ChatAPI.unsubscribeFromFriendStatus(props.friend.id, handleStatusChange);
    };
  });
  if (isOnline === null) {
    return "Загрузка...";
  }
  return isOnline ? "В сети" : "Не в сети";
}
```

- Хук находится внутри компонента для возможности использовать переменные
- если в компоненте несколько хуков, то они выполняются по очереди

## useEffect(). return и []

1.  useEffect(()=>{}) – вызывается каждый раз, когда происходит рендер
2.  useEffect(()=>{}, []) – тоже самое, что и ComponentDidMount, запустится один раз, при монтировании компонента, при последующих обновлениях запускаться не будет
3.

```js
useEffect(() => {
  return () => {
    //запустится при удалении компонента, нужен для фазы отчистки от слушателей событий, отписки и других действий при удалении компонента
  };
});
```

извлечение данных с игнорированием strict mode

```jsx
import { useState, useEffect } from 'react';
import { fetchBio } from './api.js';

export default function Page() {
  const [person, setPerson] = useState('Alice');
  const [bio, setBio] = useState(null);

  useEffect(() => {
    let ignore = false;
    setBio(null);
    fetchBio(person).then(result => {
      if (!ignore) {
        setBio(result);
      }
    });
    return () => {
      ignore = true;
    };
  }, [person]);
```

вариант с async/await

```jsx
import { useState, useEffect } from "react";
import { fetchBio } from "./api.js";

export default function Page() {
  const [person, setPerson] = useState("Alice");
  const [bio, setBio] = useState(null);
  useEffect(() => {
    async function startFetching() {
      setBio(null);
      const result = await fetchBio(person);
      if (!ignore) {
        setBio(result);
      }
    }

    let ignore = false;
    startFetching();
    return () => {
      ignore = true;
    };
  }, [person]);

  return (
    <>
      <select
        value={person}
        onChange={(e) => {
          setPerson(e.target.value);
        }}
      >
        <option value="Alice">Alice</option>
        <option value="Bob">Bob</option>
        <option value="Taylor">Taylor</option>
      </select>
      <hr />
      <p>
        <i>{bio ?? "Loading..."}</i>
      </p>
    </>
  );
}
```
