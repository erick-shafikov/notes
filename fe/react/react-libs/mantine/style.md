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

- в style-атрибуте

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
      _style={(theme) => ({
        margin: theme.spacing.xl,
        color: theme.colors.orange[5],
      })}
    >
      ...
    </Box>
  </>
);
```
