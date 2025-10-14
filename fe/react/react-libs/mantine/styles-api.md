# styles api

система классов каждого компонента, можно использовать:

# classNames

- с помощью classNames для использования css.modules, их можно использовать и в theme

```jsx
import classes from "./Demo.module.css";

const X = () => (
  <Button
    classNames={{
      inner: "my-inner-class",
      // c использованием внешних таблиц стиля
      label: classes.label,
    }}
  >
    Button
  </Button>
);
```

# styles

атрибут mantine для определения стилей по селекторам апи style-подобном синтаксисе

- атрибута styles (не желательно использовать этот вариант)

```jsx
<Button
  styles={{
    root: { backgroundColor: "red" },
    label: { color: "blue" },
    inner: { fontSize: 20 },
  }}
>
  Button
</Button>
```

# темы

Варианты переопределения в теме

```ts
const theme = createTheme({
  components: {
    Button: Button.extend({
      // если не использовать classes.module.css, а использовать только classes.css
      classNames: {
        root: "my-root-class",
        label: "my-label-class",
        inner: "my-inner-class",
      },
      styles: {
        root: { backgroundColor: "red" },
        label: { color: "blue" },
        inner: { fontSize: 20 },
      },
      // classNames может принимать _theme, props
      classNames: (_theme, props) => ({
        label: cx({ [classes.labelRequired]: props.required }),
        input: cx({ [classes.inputError]: props.error }),
      }),
    }),
  },
});
```

- все значения можно посмотреть через Component.classes
- доступ к исходным Button.classes.root

# css переменные

- css-переменные и значения для каждого компонента можно определить с помощью css модулей переопределяя значения в селекторах для классов styled-api или в теме

```jsx
import { Button, Group, MantineProvider, createTheme } from "@mantine/core";

const theme = createTheme({
  components: {
    Button: Button.extend({
      // специально поле vars, принимает функцию
      vars: (theme, props) => {
        if (props.size === "xxl") {
          return {
            root: {
              "--button-height": "60px",
              "--button-padding-x": "30px",
              "--button-fz": "24px",
            },
          };
        }

        if (props.size === "xxs") {
          return {
            root: {
              "--button-height": "24px",
              "--button-padding-x": "10px",
              "--button-fz": "10px",
            },
          };
        }

        return { root: {} };
      },
    }),
  },
});

function Demo() {
  return (
    <MantineProvider theme={theme}>
      <Group>
        <Button size="xxl">XXL Button</Button>
        <Button size="xxs">XXS Button</Button>
      </Group>
    </MantineProvider>
  );
}
```

# статичные классы

модули которые будут доступны только из classes.css, каждый компонент имеет статичный класс

```scss
.mantine-Button-root {
  background-color: red;
}
```

# атрибуты

для тестирования

```jsx
const X = () => (
  <Button
    attributes={{
      root: { "data-test-id": "root" },
      label: { "data-test-id": "label" },
      inner: { "data-test-id": "inner" },
    }}
  >
    Button
  </Button>
);
```

# data-attributes

используются для отображения состояния

```jsx
<button class="my-button" data-disabled>
  Disabled button
</button>
// <button class="my-button" data-disabled>
//   Disabled button
// </button>
```

```scss
.my-button {
  color: var(--mantine-color-black);

  &[data-disabled] {
    color: var(--mantine-color-gray-5);
  }
}
```

Варианты и размеры:

- [добавление нового варианта с помощью темы](./objects/theme.md#добавление-варианта-компонента)
- [контроль размеров через тему](#css-переменные)
- - можно и через data-атрибуты
- - через [css-переменные](#css-переменные) дял каждого стиля

## mod prop

позволяет добавить data-атрибуты

```tsx
import { Box } from "@mantine/core";

<Box mod="data-button" />;
// -> <div data-button />

<Box mod={{ opened: true }} />;
// -> <div data-opened />

<Box mod={{ opened: false }} />;
// -> <div />

<Box mod={["button", { opened: true }]} />;
// -> <div data-button data-opened />

<Box mod={{ orientation: "horizontal" }} />;
// -> <div data-orientation="horizontal" />
```
