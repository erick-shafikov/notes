<!-- Переменные ------------------------------------------------------------------------------------------------------------------------------>

# Переменные

```scss
$base-color: #c6538c;

.alert {
  border: 1px solid $base-color;
}
```

inherit работает как и со всеми свойствами, в данном случае будет lightblue

```scss
:root {
  --page-background-color: lightblue;
  background-color: tomato;
}

body {
  --page-background-color: inherit;
  background-color: var(--page-background-color);
}
```
