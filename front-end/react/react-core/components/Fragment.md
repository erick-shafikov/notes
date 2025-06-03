<!-- Fragment ------------------------------------------------------------------------------------------------------------------------------>

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
