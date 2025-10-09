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

# адаптивные стили

Варианты адаптивных стилей:

- @media
- hiddenFrom и visibleFrom

```tsx
import { Button, Group } from "@mantine/core";

const X = () => (
  <Group justify="center">
    <Button hiddenFrom="sm" color="orange">
      Hidden from sm
    </Button>
    <Button visibleFrom="sm" color="cyan">
      Visible from sm
    </Button>
    {/* в виде классов */}
    <div className="mantine-hidden-from-md">Hidden from md</div>
    <div className="mantine-visible-from-xl">Visible from xl</div>
  </Group>
);
```

- size нельзя сделать адаптивным, только через hiddenFrom и visibleFrom
- breakpoint-ы можно изменить в [теме](./objects/theme.md#breakpoints)
- c помощью хуков:
- - [useMediaQueryHook вернет значение breakpoint](./hooks/useMediaQueryHook.md)
- - [useMatches вернет настройки для компонента](./hooks/useMatches.md)
- Container queries

```scss
.child {
  @container (max-width: 500px) {
    background-color: var(--mantine-color-blue-filled);
  }

  @container (max-width: 300px) {
    background-color: var(--mantine-color-red-filled);
  }
}
```

- адаптивные styles-пропсы, со значениями base,xs, sm, md, lg, xl

```tsx
const X = () => <Box w={{ base: 320, sm: 480, lg: 640 }} />;
```

элемент будет иметь стили

```scss
.element {
  width: 20rem;
}

@media (min-width: 48em) {
  .element {
    width: 30rem;
  }
}

@media (min-width: 75em) {
  .element {
    width: 40rem;
  }
}
```
