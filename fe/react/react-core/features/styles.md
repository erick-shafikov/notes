# Поддержка таблиц стилей

Позволяет предотвратить повторную загрузку стилей

```js
function ComponentOne() {
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
  font-family: 'Popins';  
}
```

```jsx
import "./App.css";

function App() {
  return (
    <div>
      <h1>Hello</h1>
      <span className="container_span">This is a book app</span>
    </div>
  );
}
```

Модули с уникальным префиксом

```jsx
import styles from "./App.module.css";

function App() {
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
