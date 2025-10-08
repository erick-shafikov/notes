# Стилизация

Стилизация может осуществляться за счет:

- style-props - перечень пропсов, которые применимы к каждому компоненту, контролируют одно css свойство, вставляются в style атрибут
- style-prop - атрибут style
- css-modules - у каждого компонента есть собственный api

# применение темы

- в css модулях

```scss
.example {
  // theme.colors.red[5]
  background: var(--mantine-color-red-5);

  // theme.spacing.md
  margin-top: var(--mantine-spacing-md);

  // theme.headings.fontFamily
  font-family: var(--mantine-font-family-headings);
}
```

- в styles-prop

```tsx
const Ex = () => (
  <Box bg="red.5" mt="xl">
    My box
  </Box>
);
```

- в style-атрибуте может быть:
- - статичный объект стилей
- - функция, которая принимает в аргумент объект темы
- - массив

```tsx
const Ex = () => (
  <>
    <Box
      _style={{
        margin: "var(--mantine-spacing-xl)",
        color: "var(--mantine-color-orange-5)",
      }}
    >
      ...
    </Box>

    <Box
    {/* вариант с функцией */}
      _style={(theme) => ({
        margin: theme.spacing.xl,
        color: theme.colors.orange[5],
      })}
    >
      ...
    </Box>

    <Box
    {/* вариант с функцией */}
      _style={[{ color: 'red' }, style]
    >
      ...
    </Box>
  </>
);
```

# className и classNames

Возможны два варианта применения css модулей^

- classNames - специальный слоты компонентов
- className - применения ко всему
- - при выборе селектора, будут применены ко всем компонентам

```tsx
const X = () => (
  <>
    <TextInput
      classNames={{
        root: classes.root,
        input: classes.input,
        label: classes.label,
      }}
    />
  </>
);
```

# статичные стили

- mantine-active – contains :active styles
- mantine-focus-auto – contains :focus-visible styles
- mantine-focus-always – contains :focus styles
- mantine-focus-never – removes default browser focus ring
- mantine-visible-from-{breakpoint} – shows element when screen width is greater than breakpoint, for example mantine-visible-from-sm
- mantine-hidden-from-{breakpoint} – hides element when screen width is greater than breakpoint, for example mantine-hidden-from-sm

```jsx
import { Group } from "@mantine/core";

function Demo() {
  return (
    <Group>
      <button type="button" className="mantine-focus-auto">
        Focus auto
      </button>
      <button type="button" className="mantine-focus-always">
        Focus always
      </button>
      <button type="button" className="mantine-focus-never">
        Focus never
      </button>
      <button type="button" className="mantine-active">
        Active
      </button>
    </Group>
  );
}
```
