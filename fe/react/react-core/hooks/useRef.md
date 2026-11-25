# useRef

ссылка на какие либо объекты

- может передаваться в пропсах (R19)

```tsx
interface MyInputProps {
  placeholder: string;
  ref: React.Ref<HTMLInputElement>;
}

function MyInput({ placeholder, ref }: MyInputProps): JSX.Element {
  return <input placeholder={placeholder} ref={ref} />;
}

//...
<MyInput ref={ref} />;
```

- поддерживается функция отчистки

```tsx
<input
  ref={(ref: HTMLInputElement | null) => {
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

```tsx
const SomeComp = (): JSX.Element => {

  const [value, setValue] = useState<string>("initial");
const counter = useRef<number>(1); //изначальное состояние количества рендеров, возвращает объект со свойством current, которое изменяется

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

```tsx
const SomeComp = (): JSX.Element => {
  const inputRef = useRef<HTMLInputElement>(null);

  const focus = () => inputRef.current.focus();

  useEffect(() => {
    inputRef.current.value; //в inputRef.current находится текущий DOM-элемент, у которого есть тег ref
    console.log(value);
  });
  return (
    <>
      <div>{current}</div>
      <input ref={inputRef} value={value} />
      <button onClick={focus}>Фокус</button>
    </>
  );
};
```

## ref Другие применения. Получение предыдущего состояния

```tsx
import { useEffect, useState, useRef } from "react";

function UseRefHook(): JSX.Element {
  const [value, setValue] = useState<string>("initial"); //в input будет меняться состояние
  const renderCount = useRef<number>(1); //количество рендеров
  const prevValue = useRef<string>(""); //хранения предыдущего состояния

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
```

## Предотвращение рендеринга

```tsx
import { useState, useRef, useEffect } from "react";

export default function useFetch(url: string) {
  const isMounted = useRef<boolean>(false);
  const [data, setData] = useState<unknown>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState<boolean>(true);

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
export default function useFetchAll(urls: string[]) {
  const prevUrls = useRef<string[]>([]);
  const [data, setData] = useState<unknown>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    // запуститься только если массивы с url равны
    if (areEqual(prevUrls.current, urls)) return;

    prevUrls.current = urls;

    const promises = urls.map((url) =>
      fetch(process.env.REACT_APP_API_BASE_URL + url).then((response) => {
        if (response.ok) return response.json();
        throw response;
      }),
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

```tsx
const Component = () => {
  const focus = (node: HTMLDivElement) => {
    node.scrollIntoView({ behavior: "smooth" });
  };

  return <div ref={focus} />;
};
```

## ref Другие применения. Промис, который разрешается снаружи (deferred)

Задача: превратить API на колбэках или событиях (модалка, виджет, внешний скрипт) в функцию, которую можно `await`. Функция создаёт промис и кладёт его `resolve` в ref, а колбэк события позже вызывает этот `resolve` с результатом.

- `resolve` — обычная функция: промис разрешится тем значением, с которым её вызвали, где бы и когда это ни случилось
- колбэк конструктора `new Promise(...)` выполняется синхронно и сразу, промис остаётся в состоянии pending до вызова `resolve`
- ref, а не `useState`: `resolve` не нужен для отрисовки, а колбэки, созданные на прошлом рендере, всё равно видят актуальный `ref.current`

```tsx
import { useRef, useState } from "react";

type ConfirmResolver = (isConfirmed: boolean) => void;

export const useConfirm = () => {
  const [isOpen, setIsOpen] = useState(false);
  const resolverRef = useRef<ConfirmResolver | null>(null);

  const settle = (isConfirmed: boolean) => {
    resolverRef.current?.(isConfirmed);
    resolverRef.current = null;
    setIsOpen(false);
  };

  const confirm = (): Promise<boolean> => {
    settle(false);
    setIsOpen(true);

    return new Promise((resolve) => {
      resolverRef.current = resolve;
    });
  };

  const handleConfirm = () => settle(true);
  const handleCancel = () => settle(false);

  return { confirm, isOpen, handleConfirm, handleCancel };
};

const DeleteButton = ({ onDelete }: { onDelete: () => void }) => {
  const { confirm, isOpen, handleConfirm, handleCancel } = useConfirm();

  const handleClick = async () => {
    const isConfirmed = await confirm();

    if (isConfirmed) {
      onDelete();
    }
  };

  return (
    <>
      <button onClick={handleClick}>Удалить</button>
      {isOpen && (
        <div role="dialog">
          <button onClick={handleConfirm}>Да</button>
          <button onClick={handleCancel}>Нет</button>
        </div>
      )}
    </>
  );
};
```

Правила:

- разрешать промис только через одну функцию (`settle`): она вызывает `resolve` и обнуляет ref, поэтому повторный вызов ничего не делает
- каждый промис должен быть когда-нибудь разрешён: на каждый исход (успех, отмена, ошибка, повторный вызов, размонтирование) нужна ветка, иначе `await` будет ждать вечно
- повторный вызов, пока ждёт предыдущий: сначала завершить старый промис (`settle(false)`), потом положить новый `resolve`
- отложенная проверка (`setTimeout`) сравнивает `resolverRef.current` с сохранённым `resolve`, чтобы не разрешить чужой, более новый промис
- для ошибок можно хранить ещё и `reject`, но обычно проще разрешаться значением-результатом (`{ isCancelled: true }`), чтобы вызывающему не нужен был `try/catch`

С ES2024 есть `Promise.withResolvers()`, который отдаёт `{ promise, resolve, reject }` без обёртки в конструктор:

```ts
const { promise, resolve } = Promise.withResolvers<boolean>();
resolverRef.current = resolve;

return promise;
```
