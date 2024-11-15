# useCallback

Позволяет решить проблему ссылочного типа для функций, для объектов – useMemo(). Проблема – в данном компоненте, при изменении стиля, меняется рендер дочерних компонентов

```jsx
function UseCallBackHook() {
  const [colored, setColored] = useState(false);
  const [count, setCount] = useState(42);
  const styles = {
    color: colored ? "darkred" : "black",
  };
  const generateItemsFromAPI = useCallback(
    (indexNumber) => {
      //заполняем массив данными кол-во = indexNumber
      return new Array(count)
        .fill("")
        .map((_, i) => `element ${i + indexNumber}`);
    },
    [count]
  );
  return (
    <>
      <h1 _style={styles}>Amount of elements: {count}</h1>
      <button onClick={() => setCount((prev) => prev + 1)}>increase</button>
      <button
        className="btn btn-warning"
        onClick={() => setColored((prev) => !prev)}
      >
        Change
      </button>
                 
      <ItemsList getItems={generateItemsFromAPI} />
    </>
  );
}
//дочерний компонент
const  ItemsList = memo(({ getItems }) {
  const [items, setItems] = useState([]);
  useEffect(() => {
    const newItems = getItems;
    setItems(newItems);
    console.log("render");
  }, [getItems]);
  return (
    <ul>
      {items.map((i) => (
        <li key={i}>{i}</li>
      ))}
    </ul>
  );
})
```

- useCallback имеет смысл, если мы его передаем в дочерний комопнент, если дочерний обернут в memo
- если из пользовательского хука возвращаются функции, то их нужно обернуть в useCallback

псевдокод useCallback

```jsx
let store;
const useCallback = (callback, deps) => {
  if (equal(deps, store?.deps)) return store.callback;

  store = { deps, callback };
  return callback;
};

const Component = () => {
  const handleChange = useCallback(() => {}, []);
};
```

## BP. useCallback как альтернатива ref

```jsx
const Comp = () => {
  const setFocus = useCallback((element) => {
    element.focus();
  }, []);

  return <input ref={setFocus()} />;
};
```

## BP. мемоизация списка инпутов

```tsx
import React, { FC, useState, memo, useCallback, useRef } from "react";

export const genId = (items: { key: string }[]): string => {
  if (!items?.length) return "1";
  return (Math.max(...items?.map((c) => parseInt(c.key, 10))) + 1).toString();
};

export type StringInputProps = {
  id?: string;
  value: string;
  onChange: (value: string, id?: string) => void;
};

export const StringInput = memo<StringInputProps>(({ value, onChange, id }) => {
  console.log("rerender StringInput");
  return <input value={value} onChange={(e) => onChange(e.target.value, id)} />;
});

type StringsInputItem = {
  key: string;
  value: string;
};

export type StringsInputProps = {
  value: StringsInputItem[];
  onChange: (value: StringsInputItem[]) => void;
};

export const StringsInput: FC<StringsInputProps> = ({ value, onChange }) => {
  const onAdd = () => {
    const newValue = [...(value || [])];
    newValue.push({ key: genId(value), value: undefined });
    onChange(newValue);
  };
  const valueCopy = useRef(value);
  valueCopy.current = value;

  const handleChange = useCallback(
    (_value: string, id?: string) => {
      const newValue = (valueCopy.current || []).map((i) =>
        i.key === id ? { key: id, value: _value } : i
      );
      onChange(newValue);
    },
    [onChange]
  );

  return (
    <div>
      {value?.map((item) => {
        return (
          <div key={item.key}>
            <StringInput
              id={item.key}
              key={item.key}
              value={item.value}
              onChange={handleChange}
            />
            <button
              type="button"
              onClick={() => onChange(value?.filter((i) => i.key !== item.key))}
            >
              -
            </button>
          </div>
        );
      })}
      <div>
        <button type="button" onClick={onAdd}>
          +
        </button>
      </div>
    </div>
  );
};

export const ExampleEventSwitch: FC = () => {
  const [value, setValue] = useState<StringsInputItem[]>();
  return (
    <StringsInput value={value as StringsInputItem[]} onChange={setValue} />
  );
};
```

Вариант 2

```tsx
import React, { FC, useState, memo, useMemo, useRef } from "react";

export const genId = (items: { key: string }[]): string => {
  if (!items?.length) return "1";
  return (Math.max(...items?.map((c) => parseInt(c.key, 10))) + 1).toString();
};

export type StringInputProps = {
  value: string;
  onChange: (value: string) => void;
};

export const StringInput = memo<StringInputProps>(({ value, onChange }) => {
  console.log("rerender StringInput");
  return <input value={value} onChange={(e) => onChange(e.target.value)} />;
});

type StringsInputItem = {
  key: string;
  value: string;
};

export type StringsInputProps = {
  value: StringsInputItem[];
  onChange: (value: StringsInputItem[]) => void;
};

export const StringsInput: FC<StringsInputProps> = ({ value, onChange }) => {
  const onAdd = () => {
    const newValue = [...(value || [])];
    newValue.push({ key: genId(value), value: undefined });
    onChange(newValue);
  };
  const valueCopy = useRef(value);
  valueCopy.current = value;

  const handleChange = useMemo(() => {
    const cache: Record<string, (value: string) => void> = {};
    return (id: string) => {
      if (cache[id]) return cache[id];
      cache[id] = (_value: string) => {
        const newValue = (valueCopy.current || []).map((i) =>
          i.key === id ? { key: id, value: _value } : i
        );
        onChange(newValue);
      };
      return cache[id];
    };
  }, [onChange]);

  return (
    <div>
      {value?.map((item) => {
        return (
          <div key={item.key}>
            <StringInput value={item.value} onChange={handleChange(item.key)} />
            <button
              type="button"
              onClick={() => onChange(value?.filter((i) => i.key !== item.key))}
            >
              -
            </button>
          </div>
        );
      })}
      <div>
        <button type="button" onClick={onAdd}>
          +
        </button>
      </div>
    </div>
  );
};

export const ExampleCaching: FC = () => {
  const [value, setValue] = useState<StringsInputItem[]>();
  return (
    <StringsInput value={value as StringsInputItem[]} onChange={setValue} />
  );
};
```

# BP. Event switch

```tsx
const AddressInput = ({ value, onChange }) => {
  const handleChange = (e) =>
    onChange({ ...value, [e.target.name]: e.target.value });
  return (
    <div>
      <div>
        <div>city</div>
        <input name="city" value={value?.city} onChange={handleChange} />
      </div>
      <div>
        <div>street</div>
        <input name="street" value={value?.street} onChange={handleChange} />
      </div>
      <div>
        <div>house</div>
        <input name="house" value={value?.house} onChange={handleChange} />
      </div>
    </div>
  );
};
```
