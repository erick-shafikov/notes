# РџРѕРґРґРµСЂР¶РєР° С‚Р°Р±Р»РёС† СЃС‚РёР»РµР№

РџРѕР·РІРѕР»СЏРµС‚ РїСЂРµРґРѕС‚РІСЂР°С‚РёС‚СЊ РїРѕРІС‚РѕСЂРЅСѓСЋ Р·Р°РіСЂСѓР·РєСѓ СЃС‚РёР»РµР№

```tsx
function ComponentOne(): JSX.Element {
  return (
    <Suspense fallback="loading...">
      <link rel="stylesheet" href="foo" precedence="default" />
      <link rel="stylesheet" href="bar" precedence="high" />
      <article class="foo-class bar-class">
        {...}
      </article>
    </Suspense>
  )
}

function ComponentTwo() {
  return (
    <div>
      <p>{...}</p>
      <link rel="stylesheet" href="baz" precedence="default" />  <-- will be inserted between foo & bar
    </div>
  )
}
```

# modules

```scss
.container_span {
  text-shadow: 4px 4px 3px rgb(154, 158, 161);
  text-decoration: underline;
  font-family: 'Popins';
}

h1{
  font-family: 'Popins'; В 
}
```

```tsx
import "./App.css";

function App(): JSX.Element {
  return (
    <div>
      <h1>Hello</h1>
      <span className="container_span">This is a book app</span>
    </div>
  );
}
```

РњРѕРґСѓР»Рё СЃ СѓРЅРёРєР°Р»СЊРЅС‹Рј РїСЂРµС„РёРєСЃРѕРј

```tsx
import styles from "./App.module.css";

function App(): JSX.Element {
  return (
    <div>
      <h1>Hello</h1>
      <span className={styles.container_span}>This is a book app</span>
    </div>
  );
}
```

```scss
.green {
  border: 1px solid green;
}
.input.red {
  border: 1px solid red;
}
```
