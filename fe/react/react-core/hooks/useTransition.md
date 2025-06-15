## useTransition

```js
const [isPending, startTransition] = useTransition();
```

обновляет значение состояния без блокирования UI, isPending – отображает статус обновляется ли Ui, startTransition – функция для обновления, которая дает низкий приоритет функции с изменением состояния в CC-режиме. Лучше вызывать ниже в дереве компонентов. isPending === true если startTransition не выполняется

useTransition – позволяет обновить state без блокировки UI, устанавливает сеттеру состояния, преданный в callback, более низкий приоритет, при стадии рендера
isPending – флаг, который показывает находится ли в ожидании переход, startTransition – запуск перехода по смене состояния

```jsx
import { useState, useTransition } from "react";
import TabButton from "./TabButton.js";
import AboutTab from "./AboutTab.js";
import PostsTab from "./PostsTab.js";
import ContactTab from "./ContactTab.js";

export default function TabContainer() {
  const [isPending, startTransition] = useTransition();
  const [tab, setTab] = useState("about");

  function selectTab(nextTab) {
    startTransition(() => {
      setTab(nextTab);
    });
  }

  return (
    <>
      <TabButton isActive={tab === "about"} onClick={() => selectTab("about")}>
        About
      </TabButton>
      <TabButton isActive={tab === "posts"} onClick={() => selectTab("posts")}>
        Posts (slow)
      </TabButton>
      <TabButton
        isActive={tab === "contact"}
        onClick={() => selectTab("contact")}
      >
        Contact
      </TabButton>
      <hr />
      {tab === "about" && <AboutTab />}
      {tab === "posts" && <PostsTab />}
      {tab === "contact" && <ContactTab />}
    </>
  );
}
```

Могут быть использованы и асинхронные функции в качестве callback

```js
function UpdateName({}) {
  const [name, setName] = useState("");
  const [error, setError] = useState(null);
  const [isPending, startTransition] = useTransition();

  const handleSubmit = () => {
    startTransition(async () => {
      const error = await updateName(name);
      if (error) {
        setError(error);
        return;
      }
      redirect("/path");
    });
  };

  return (
    <div>
      <input value={name} onChange={(event) => setName(event.target.value)} />
      <button onClick={handleSubmit} disabled={isPending}>
        Update
      </button>
      {error && <p>{error}</p>}
    </div>
  );
}
```
