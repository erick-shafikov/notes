# Fragment

при нарушении семантической верстки из-за тэгов `<div>` могут возникать проблемы

```js
import React, { Fragment } from "react";
function ListItem({ item }) {
  return (
    <Fragment>
      <dt>{item.term}</dt>
      <dd>{item.description}</dd> {" "}
    </Fragment>
  );
}
function Glossary(props) {
  return (
    <dl>
      {props.items.map((item) => (
        <ListItem item={item} key={item.id} />
      ))}
    </dl>
  );
}

function ListItem({ item }) {
  return (
    <>
      <dt>{item.term}</dt>
      <dd>{item.description}</dd>
    </>
  );
}
```

## Key для фрагмента

```tsx
<React.Fragment key="a"></React.Fragment> - можно добавить key
<></> - нельзя добавить key
```

## ref для фрагмента (React 19.3)

`<Fragment ref={...}>` позволяет работать с группой DOM-узлов без обёртки `<div>`. `ref.current` — это `FragmentInstance`.

**Методы `FragmentInstance`:**

- `addEventListener`, `removeEventListener`, `dispatchEvent` — управление событиями для детей первого уровня
- `focus`, `focusLast`, `blur` — перемещение фокуса по вложенным детям (depth-first)
- `observeUsing`, `unobserveUsing` — подключение `IntersectionObserver` или `ResizeObserver`
- `getClientRects`, `getRootNode`, `compareDocumentPosition`, `scrollIntoView` — измерение и прокрутка

```jsx
import { Fragment, useRef, useLayoutEffect } from "react";

export default function InView({ onChange, children }) {
  const fragmentRef = useRef(null);

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
