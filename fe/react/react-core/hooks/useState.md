# useState

РђСЃРёРЅС…СЂРѕРЅРЅС‹Рµ С…СѓРє
РЅРµР»СЊР·СЏ РІС‹Р·С‹РІР°С‚СЊ РІ Р»РѕРіРёС‡РµСЃРєРёС… Р±Р»РѕРєР°С…

```tsx
function Example(): JSX.Element {
  // РћР±СЉСЏРІР»СЏРµРј РЅРѕРІСѓСЋ РїРµСЂРµРјРµРЅРЅСѓСЋ СЃРѕСЃС‚РѕСЏРЅРёСЏ "count"
  const [count, setCount] = useState<number>(0); // state: {count:0}
  return (
    <div>
      <p>Р’С‹ РЅР°Р¶Р°Р»Рё {count} СЂР°Р·</p>
      <button onClick={() => setCount(count + 1)}>РќР°Р¶РјРё РЅР° РјРµРЅСЏ</button>
    </div>
  );
}
```

useState РІРѕР·РІСЂР°С‰Р°РµС‚ РјР°СЃСЃРёРІ СЃ РґРІСѓРјСЏ СЌР»РµРјРµРЅС‚Р°РјРё вЂ“ С‚РµРєСѓС‰РµРµ Р·РЅР°С‡РµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ Рё С„СѓРЅРєС†РёСЋ РґР»СЏ РµРіРѕ РёР·РјРµРЅРµРЅРёСЏ
РµРґРёРЅСЃС‚РІРµРЅРЅС‹Р№ Р°СЂРіСѓРјРµРЅС‚ вЂ“ РёР·РЅР°С‡Р°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ

- РѕР±СЉСЏРІР»СЏС‚СЊ setState РІРЅСѓС‚СЂРё СѓСЃР»РѕРІРЅС‹С… Р±Р»РѕРєРѕРІ РЅРµР»СЊР·СЏ
- РµСЃР»Рё РґРµР±Р°Р¶РёС‚СЊ СЃРѕСЃС‚РѕСЏРЅРёРµ РїРѕСЃР»Рµ РёР·РјРµРЅРµРЅРёСЏ, С‚Рѕ РѕРЅРѕ Р±СѓРґРµС‚ РїСЂРµРґС‹РґСѓС‰РµРµ

```tsx
function ExampleWithManyStates(): null {
  // РћР±СЉСЏРІР»СЏРµРј РЅРµСЃРєРѕР»СЊРєРѕ РїРµСЂРµРјРµРЅРЅС‹С… СЃРѕСЃС‚РѕСЏРЅРёСЏ!
  const [age, setAge] = useState<number>(42);
  const [fruit, setFruit] = useState<string>("Р±Р°РЅР°РЅ");
  const [todos, setTodos] = useState<{ text: string }[]>([{ text: "РР·СѓС‡РёС‚СЊ С…СѓРєРё" }]); // ...
  return null;
}
```

## prevValue

Р¤РѕСЂРјР°, РєРѕС‚РѕСЂР°СЏ РґРёРЅР°РјРёС‡РµСЃРєРё РѕС‚РѕР±СЂР°Р¶Р°РµС‚ РёР·РјРµРЅРµРЅРёСЏ РІ РґРІСѓС… РїРѕР»СЏС…

```tsx
interface FullName {
  fName: string;
  lName: string;
}

function App(): null {
  const [fullName, setFullName] = useState<FullName>({
    //РёСЃС…РѕРґРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ
    fName: "",
    lName: "",
  });
  const changeHandler = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value, name } = e.target; //РґРµСЃС‚СЂСѓРєС‚СѓСЂРёР·Р°С†РёСЏ РѕР±СЉРµРєС‚Р° e.target, С‚Р°Рє РєР°Рє РѕРґРёРЅР°РєРѕРІС‹Рµ РѕР±СЂР°Р±РѕС‚С‡РёРєРё СЃРѕР±С‹С‚РёСЏ (changeHandler) РѕС‚РЅРѕСЃСЏС‚СЃСЏ Рє РґРІСѓРј РїРѕР»СЏРј С„РѕСЂРјС‹, С‚Рѕ target.name Р±СѓРґРµС‚ РѕРїСЂРµРґРµР»СЏС‚СЊ РєР°РєРѕРµ РїРѕР»Рµ РёР·РјРµРЅСЏРµС‚СЃСЏ

    setFullName((prevValue) => {
      //prevValue вЂ“ СЌС‚Рѕ Р°СЂРіСѓРјРµРЅС‚ С„СѓРЅРєС†РёРё, РєРѕС‚РѕСЂС‹Р№ С…Р°СЂР°РєС‚РµСЂРёР·СѓРµС‚ СЃРѕР±РѕР№ РїСЂРµРґС‹РґСѓС‰РµРµ СЃРѕСЃС‚РѕСЏРЅРёРµ, РЅРёР¶Рµ РІРѕР·РІСЂР°С‰Р°СЋС‚СЃСЏ РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ СЃРёС‚СѓР°С†РёРё РґРІР° РІР°СЂРёР°РЅС‚Р° РѕР±СЉРµРєС‚Р° СЃРѕСЃС‚РѕСЏРЅРёСЏ
      if (name === "fName") {
        return {
          fName: value, //РїРµСЂРµРїРёСЃС‹РІР°РµРј
          lName: prevValue.lName, //СЃРѕС…СЂР°РЅСЏРµРј СЃС‚Р°СЂРѕРµ Р·РЅР°С‡РµРЅРёРµ
        };
      } else {
        return {
          fName: prevValue.fName,
          lName: value,
        };
      }
    });
  };
  return null;
}
```

Р°Р»СЊС‚РµСЂРЅР°С‚РёРІР° РґРІСѓРј if

```tsx
function App(): null {
  const [fullName, setFullName] = useState<FullName>({
    //РёСЃС…РѕРґРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ
    fName: "",
    lName: "",
  });
  const changeHandler = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value, name } = e.target; //РґРµСЃС‚СЂСѓРєС‚СѓСЂРёР·Р°С†РёСЏ РѕР±СЉРµРєС‚Р° e.target, С‚Р°Рє РєР°Рє РѕРґРёРЅР°РєРѕРІС‹Рµ РѕР±СЂР°Р±РѕС‚С‡РёРєРё СЃРѕР±С‹С‚РёСЏ (changeHandler) РѕС‚РЅРѕСЃСЏС‚СЃСЏ Рє РґРІСѓРј РїРѕР»СЏРј С„РѕСЂРјС‹, С‚Рѕ target.name Р±СѓРґРµС‚ РѕРїСЂРµРґРµР»СЏС‚СЊ РєР°РєРѕРµ РїРѕР»Рµ РёР·РјРµРЅСЏРµС‚СЃСЏ
    setFullName((prevValue) => {
      return {
        ...prevValue,
        [name]: value,
      };
    });
  };
  return null;
}
```

РЎРѕР·РґР°РЅРёРµ РїСЂРµРґС‹РґСѓС‰РµРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ

```tsx
// РЅРµ РѕРє
const [todos, setTodos] = useState(createInitialTodos());
// РѕРє
const [todos, setTodos] = useState(createInitialTodos);
```

## BP

```tsx
import React, { useState } from "react";
import ToDoItem from "./ToDoItem";
function App(): JSX.Element {
  const [inputText, setInputText] = useState<string>("");
  const [items, setItems] = useState<string[]>([]);
  function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    const newValue = event.target.value;
    setInputText(newValue);
  }
  function addItem() {
    setItems((prevItems) => {
      return [...prevItems, inputText];
    });
    setInputText("");
  }
  function deleteItem(id: number) {
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
        <input onChange={handleChange} type="text" value={inputText} />В 
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
