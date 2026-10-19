# Fragment

РїСЂРё РЅР°СЂСѓС€РµРЅРёРё СЃРµРјР°РЅС‚РёС‡РµСЃРєРѕР№ РІРµСЂСЃС‚РєРё РёР·-Р·Р° С‚СЌРіРѕРІ `<div>` РјРѕРіСѓС‚ РІРѕР·РЅРёРєР°С‚СЊ РїСЂРѕР±Р»РµРјС‹

```tsx
import React, { Fragment } from "react";
function ListItem({ item }: { item: { id: string; term: string; description: string } }): JSX.Element {
  return (
    <Fragment>
      <dt>{item.term}</dt>
      <dd>{item.description}</dd>В {" "}
    </Fragment>
  );
}
function Glossary({ items }: { items: { id: string; term: string; description: string }[] }): JSX.Element {
  return (
    <dl>
      {items.map((item) => (
        <ListItem item={item} key={item.id} />
      ))}
    </dl>
  );
}

function ListItem({ item }: { item: { id: string; term: string; description: string } }): JSX.Element {
  return (
    <>
      <dt>{item.term}</dt>
      <dd>{item.description}</dd>
    </>
  );
}
```

## Key РґР»СЏ С„СЂР°РіРјРµРЅС‚Р°

```tsx
<React.Fragment key="a"></React.Fragment> - РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ key
<></> - РЅРµР»СЊР·СЏ РґРѕР±Р°РІРёС‚СЊ key
```

## ref РґР»СЏ С„СЂР°РіРјРµРЅС‚Р° (React 19.3)

`<Fragment ref={...}>` РїРѕР·РІРѕР»СЏРµС‚ СЂР°Р±РѕС‚Р°С‚СЊ СЃ РіСЂСѓРїРїРѕР№ DOM-СѓР·Р»РѕРІ Р±РµР· РѕР±С‘СЂС‚РєРё `<div>`. `ref.current` вЂ” СЌС‚Рѕ `FragmentInstance`.

**РњРµС‚РѕРґС‹ `FragmentInstance`:**

- `addEventListener`, `removeEventListener`, `dispatchEvent` вЂ” СѓРїСЂР°РІР»РµРЅРёРµ СЃРѕР±С‹С‚РёСЏРјРё РґР»СЏ РґРµС‚РµР№ РїРµСЂРІРѕРіРѕ СѓСЂРѕРІРЅСЏ
- `focus`, `focusLast`, `blur` вЂ” РїРµСЂРµРјРµС‰РµРЅРёРµ С„РѕРєСѓСЃР° РїРѕ РІР»РѕР¶РµРЅРЅС‹Рј РґРµС‚СЏРј (depth-first)
- `observeUsing`, `unobserveUsing` вЂ” РїРѕРґРєР»СЋС‡РµРЅРёРµ `IntersectionObserver` РёР»Рё `ResizeObserver`
- `getClientRects`, `getRootNode`, `compareDocumentPosition`, `scrollIntoView` вЂ” РёР·РјРµСЂРµРЅРёРµ Рё РїСЂРѕРєСЂСѓС‚РєР°

```tsx
import { Fragment, useRef, useLayoutEffect } from "react";

interface InViewProps {
  onChange: (visible: boolean) => void;
  children: React.ReactNode;
}

export default function InView({ onChange, children }: InViewProps): JSX.Element {
  const fragmentRef = useRef<any>(null);

  useLayoutEffect(() => {
    const visibleElements = new Set();
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) visibleElements.add(e.target);
        else visibleElements.delete(e.target);
      });
      onChange(visibleElements.size > 0);
    });

    const fragmentInstance = fragmentRef.current;
    fragmentInstance.observeUsing(observer);
    return () => fragmentInstance.unobserveUsing(observer);
  }, [onChange]);

  return <Fragment ref={fragmentRef}>{children}</Fragment>;
}
```
