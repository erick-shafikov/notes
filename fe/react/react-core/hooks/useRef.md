# useRef

ссылка на какие либо объекты

- может передаваться в пропсах (R19)

```js
function MyInput({ placeholder, ref }) {
  return <input placeholder={placeholder} ref={ref} />;
}

//...
<MyInput ref={ref} />;
```

- поддерживается функция отчистки

```js
<input
  ref={(ref) => {
    // ref создан

    // НОВОЕ: возврат функции очистки для сброса
    // ссылка, когда элемент удаляется из DOM.
    return () => {
      // очистка ref
    };
  }
/>
```

## ref Другие применения. подсчет количества рендеров

```jsx

const SomeComp = () => {

  const [value, setValue] = useState("initial");
const counter = useRef(1); //изначальное состояние количества рендеров, возвращает объект со свойством current, которое изменяется

  useEffect(()=>{//запускается так как меняется состояние компонента
    counter.current ++ //увеличение при рендере, так как ниже в <input> идет ввод символов, который вызывает повторный рендер компонентов
  });

  return(
    <div>{counter.current}</div>
    <input onChange={e => setValue(prev => e.target.value)} value={value} />
  )
}

```

## ref Другие применения. Получение DOM-элемента

```jsx
const SomeComp =() => {

  const inputRef = useRef();


  const focus = () = inputRef.current.focus();


  useEffect(()=>{
    inputRef.current.value;//в inputRef.current находится текущий DOM-элемент, у которого есть тег ref
    console.log(value);
  });
  return(
    <div>{current}</div>
    <input ref={inputRef} value={value}>
    <button onClick={focus}>Фокус</button>

  )
}

```

## ref Другие применения. Получение предыдущего состояния

```js
import { useEffect, useState, useRef } from "react";
function UseRefHook() {
  const [value, setValue] = useState("initial"); //в input будет меняться состояние
  const renderCount = useRef(1); //количество рендеров
  const prevValue = useRef(""); //хранения предыдущего состояния

  useEffect(() => {
    renderCount.current++; //счетчик количества рендеров
  });

  useEffect(() => {
    prevValue.current = value; //вызывать при изменения значения в input значения value
  }, [value]);

  return (
    <div>
      <h1>Previous state{prevValue.current}</h1>
      <h1>Mount of render {renderCount.current}</h1>
      <input
        type="text"
        onChange={(e) => setValue(e.target.value)}
        value={value}
      ></input>        
    </div>
  );
}
export default UseRefHook;
```

## Предотвращение рендеринга

```js
import { useState, useRef, useEffect } from "react";

export default function useFetch(url) {
  const isMounted = useRef(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    isMounted.current = true;
    async function init() {
      try {
        const response = await fetch();
        if (response.ok) {
          const json = await response.json();
          if (isMounted.current) setData(json);
        } else {
          throw response;
        }
      } catch (e) {
        if (isMounted.current) setError(e);
      } finally {
        if (isMounted.current) setLoading(false);
      }
    }
    init();

    return () => {
      isMounted.current = false;
    };
  }, [url]);

  return { data, error, loading };
}

// второй хук ссылающийся на параметры
export default function useFetchAll(urls) {
  const prevUrls = useRef([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // запуститься только если массивы с url равны
    if (areEqual(prevUrls.current, urls)) return;

    prevUrls.current = urls;

    const promises = urls.map((url) =>
      fetch(process.env.REACT_APP_API_BASE_URL + url).then((response) => {
        if (response.ok) return response.json();
        throw response;
      })
    );

    Promise.all(promises)
      .then((json) => setData(json))
      .catch((e) => {
        console.error(e);
        setError(e);
      })
      .finally(() => setLoading(false));
  }, [urls]);

  return { data, loading, error };
}

// утилита по сравнению двух массивов
function areEqual(array1, array2) {
  return (
    array1.length === array2.length &&
    array1.every((value, index) => value === array2[index])
  );
}
```

# bp. автофокус

```tsx
const Ref = () => <input ref={(node) => node && node.focus()} />;
```
