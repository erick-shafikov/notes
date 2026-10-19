# useCallback

РџРѕР·РІРѕР»СЏРµС‚ СЂРµС€РёС‚СЊ РїСЂРѕР±Р»РµРјСѓ СЃСЃС‹Р»РѕС‡РЅРѕРіРѕ С‚РёРїР° РґР»СЏ С„СѓРЅРєС†РёР№, РґР»СЏ РѕР±СЉРµРєС‚РѕРІ вЂ“ useMemo(). РџСЂРѕР±Р»РµРјР° вЂ“ РІ РґР°РЅРЅРѕРј РєРѕРјРїРѕРЅРµРЅС‚Рµ, РїСЂРё РёР·РјРµРЅРµРЅРёРё СЃС‚РёР»СЏ, РјРµРЅСЏРµС‚СЃСЏ СЂРµРЅРґРµСЂ РґРѕС‡РµСЂРЅРёС… РєРѕРјРїРѕРЅРµРЅС‚РѕРІ

```tsx
function UseCallBackHook(): JSX.Element {
  const [colored, setColored] = useState<boolean>(false);
  const [count, setCount] = useState<number>(42);
  const styles = {
    color: colored ? "darkred" : "black",
  };
  const generateItemsFromAPI = useCallback(
    (indexNumber: number): string[] => {
      //Р·Р°РїРѕР»РЅСЏРµРј РјР°СЃСЃРёРІ РґР°РЅРЅС‹РјРё РєРѕР»-РІРѕ = indexNumber
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
      В  В  В  В  В  В 
      <ItemsList getItems={generateItemsFromAPI} />
    </>
  );
}
//РґРѕС‡РµСЂРЅРёР№ РєРѕРјРїРѕРЅРµРЅС‚
const ItemsList = memo(({ getItems }: { getItems: (i: number) => string[] }): JSX.Element => {
  const [items, setItems] = useState<string[]>([]);
  useEffect(() => {
    const newItems = getItems(0);
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

- useCallback РёРјРµРµС‚ СЃРјС‹СЃР», РµСЃР»Рё РјС‹ РµРіРѕ РїРµСЂРµРґР°РµРј РІ РґРѕС‡РµСЂРЅРёР№ РєРѕРјРѕРїРЅРµРЅС‚, РµСЃР»Рё РґРѕС‡РµСЂРЅРёР№ РѕР±РµСЂРЅСѓС‚ РІ memo
- РµСЃР»Рё РёР· РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРѕРіРѕ С…СѓРєР° РІРѕР·РІСЂР°С‰Р°СЋС‚СЃСЏ С„СѓРЅРєС†РёРё, С‚Рѕ РёС… РЅСѓР¶РЅРѕ РѕР±РµСЂРЅСѓС‚СЊ РІ useCallback

РїСЃРµРІРґРѕРєРѕРґ useCallback

```tsx
let store: { deps: unknown[]; callback: Function } | undefined;
const useCallback = (callback, deps) => {
  if (equal(deps, store?.deps)) return store.callback;

  store = { deps, callback };
  return callback;
};

const Component = () => {
  const handleChange = useCallback(() => {}, []);
};
```

## BP. useCallback РєР°Рє Р°Р»СЊС‚РµСЂРЅР°С‚РёРІР° ref

```tsx
const Comp = (): JSX.Element => {
  const setFocus = useCallback((element) => {
    element.focus();
  }, []);

  return <input ref={setFocus()} />;
};
```

## BP. РјРµРјРѕРёР·Р°С†РёСЏ СЃРїРёСЃРєР° РёРЅРїСѓС‚РѕРІ

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

Р’Р°СЂРёР°РЅС‚ 2

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
```

```tsx
export const ExampleCaching: FC = () => {
  const [value, setValue] = useState<StringsInputItem[]>();
  return (
    <StringsInput value={value as StringsInputItem[]} onChange={setValue} />
  );
};
```


