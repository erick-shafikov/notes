# useRef

СЃСЃС‹Р»РєР° РЅР° РєР°РєРёРµ Р»РёР±Рѕ РѕР±СЉРµРєС‚С‹

- РјРѕР¶РµС‚ РїРµСЂРµРґР°РІР°С‚СЊСЃСЏ РІ РїСЂРѕРїСЃР°С… (R19)

```tsx
interface MyInputProps \{
  placeholder: string;
  ref: React.Ref<HTMLInputElement>;
\}

function MyInput(\{ placeholder, ref \}: MyInputProps): JSX.Element {
  return <input placeholder={placeholder} ref={ref} />;
}

//...
<MyInput ref={ref} />;
```

- РїРѕРґРґРµСЂР¶РёРІР°РµС‚СЃСЏ С„СѓРЅРєС†РёСЏ РѕС‚С‡РёСЃС‚РєРё

```tsx
<input
  ref={(ref: HTMLInputElement | null) => {
    // ref СЃРѕР·РґР°РЅ

    // РќРћР’РћР•: РІРѕР·РІСЂР°С‚ С„СѓРЅРєС†РёРё РѕС‡РёСЃС‚РєРё РґР»СЏ СЃР±СЂРѕСЃР°
    // СЃСЃС‹Р»РєР°, РєРѕРіРґР° СЌР»РµРјРµРЅС‚ СѓРґР°Р»СЏРµС‚СЃСЏ РёР· DOM.
    return () => {
      // РѕС‡РёСЃС‚РєР° ref
    };
  }
/>
```

## ref Р”СЂСѓРіРёРµ РїСЂРёРјРµРЅРµРЅРёСЏ. РїРѕРґСЃС‡РµС‚ РєРѕР»РёС‡РµСЃС‚РІР° СЂРµРЅРґРµСЂРѕРІ

```tsx
const SomeComp = (): JSX.Element => {

  const [value, setValue] = useState<string>("initial");
const counter = useRef<number>(1); //РёР·РЅР°С‡Р°Р»СЊРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ РєРѕР»РёС‡РµСЃС‚РІР° СЂРµРЅРґРµСЂРѕРІ, РІРѕР·РІСЂР°С‰Р°РµС‚ РѕР±СЉРµРєС‚ СЃРѕ СЃРІРѕР№СЃС‚РІРѕРј current, РєРѕС‚РѕСЂРѕРµ РёР·РјРµРЅСЏРµС‚СЃСЏ

  useEffect(()=>{//Р·Р°РїСѓСЃРєР°РµС‚СЃСЏ С‚Р°Рє РєР°Рє РјРµРЅСЏРµС‚СЃСЏ СЃРѕСЃС‚РѕСЏРЅРёРµ РєРѕРјРїРѕРЅРµРЅС‚Р°
    counter.current ++ //СѓРІРµР»РёС‡РµРЅРёРµ РїСЂРё СЂРµРЅРґРµСЂРµ, С‚Р°Рє РєР°Рє РЅРёР¶Рµ РІ <input> РёРґРµС‚ РІРІРѕРґ СЃРёРјРІРѕР»РѕРІ, РєРѕС‚РѕСЂС‹Р№ РІС‹Р·С‹РІР°РµС‚ РїРѕРІС‚РѕСЂРЅС‹Р№ СЂРµРЅРґРµСЂ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ
  });

  return(
    <div>{counter.current}</div>
    <input onChange={e => setValue(prev => e.target.value)} value={value} />
  )
}

```

## ref Р”СЂСѓРіРёРµ РїСЂРёРјРµРЅРµРЅРёСЏ. РџРѕР»СѓС‡РµРЅРёРµ DOM-СЌР»РµРјРµРЅС‚Р°

```tsx
const SomeComp = (): JSX.Element => {

  const inputRef = useRef<HTMLInputElement>(null);


  const focus = () => inputRef.current.focus();


  useEffect(()=>{
    inputRef.current.value;//РІ inputRef.current РЅР°С…РѕРґРёС‚СЃСЏ С‚РµРєСѓС‰РёР№ DOM-СЌР»РµРјРµРЅС‚, Сѓ РєРѕС‚РѕСЂРѕРіРѕ РµСЃС‚СЊ С‚РµРі ref
    console.log(value);
  });
  return(
    <div>{current}</div>
    <input ref={inputRef} value={value}>
    <button onClick={focus}>Р¤РѕРєСѓСЃ</button>

  )
}

```

## ref Р”СЂСѓРіРёРµ РїСЂРёРјРµРЅРµРЅРёСЏ. РџРѕР»СѓС‡РµРЅРёРµ РїСЂРµРґС‹РґСѓС‰РµРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ

```tsx
import { useEffect, useState, useRef } from "react";
function UseRefHook(): JSX.Element {
  const [value, setValue] = useState<string>("initial"); //РІ input Р±СѓРґРµС‚ РјРµРЅСЏС‚СЊСЃСЏ СЃРѕСЃС‚РѕСЏРЅРёРµ
  const renderCount = useRef<number>(1); //РєРѕР»РёС‡РµСЃС‚РІРѕ СЂРµРЅРґРµСЂРѕРІ
  const prevValue = useRef<string>(""); //С…СЂР°РЅРµРЅРёСЏ РїСЂРµРґС‹РґСѓС‰РµРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ

  useEffect(() => {
    renderCount.current++; //СЃС‡РµС‚С‡РёРє РєРѕР»РёС‡РµСЃС‚РІР° СЂРµРЅРґРµСЂРѕРІ
  });

  useEffect(() => {
    prevValue.current = value; //РІС‹Р·С‹РІР°С‚СЊ РїСЂРё РёР·РјРµРЅРµРЅРёСЏ Р·РЅР°С‡РµРЅРёСЏ РІ input Р·РЅР°С‡РµРЅРёСЏ value
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
      В  В  В  В В 
    </div>
  );
}
export default UseRefHook;
```

## РџСЂРµРґРѕС‚РІСЂР°С‰РµРЅРёРµ СЂРµРЅРґРµСЂРёРЅРіР°

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

// РІС‚РѕСЂРѕР№ С…СѓРє СЃСЃС‹Р»Р°СЋС‰РёР№СЃСЏ РЅР° РїР°СЂР°РјРµС‚СЂС‹
export default function useFetchAll(urls: string[]) {
  const prevUrls = useRef<string[]>([]);
  const [data, setData] = useState<unknown>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    // Р·Р°РїСѓСЃС‚РёС‚СЊСЃСЏ С‚РѕР»СЊРєРѕ РµСЃР»Рё РјР°СЃСЃРёРІС‹ СЃ url СЂР°РІРЅС‹
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

// СѓС‚РёР»РёС‚Р° РїРѕ СЃСЂР°РІРЅРµРЅРёСЋ РґРІСѓС… РјР°СЃСЃРёРІРѕРІ
function areEqual(array1, array2) {
  return (
    array1.length === array2.length &&
    array1.every((value, index) => value === array2[index])
  );
}
```

# bp. Р°РІС‚РѕС„РѕРєСѓСЃ

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
