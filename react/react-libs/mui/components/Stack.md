# Stack

Компонент для вертикально расположенных элементов

```tsx
<Stack
  spacing={2}
  divider={<Divider orientation="vertical" flexItem />}
  direction={{ xs: "column", sm: "row" }}
  spacing={{ xs: 1, sm: 2, md: 4 }}
>
  <Item>Item 1</Item>
  <Item>Item 2</Item>
  <Item>Item 3</Item>
</Stack>
```

```ts
type StackProps = {
  children: node;
  component: elementType;
  sx: Array<func | object | bool> | func | object;
  // --------------------------------------------------------------------
  direction:
    | "column-reverse"
    | "column"
    | "row-reverse"
    | "row"
    | Array<"column-reverse" | "column" | "row-reverse" | "row">
    | object; //направление { xs: "column"; sm: "row" }
  divider: ReactNode; //разделитель
  spicing: number | object; //{ xs: 1; sm: 2; md: 4; ... };
  useFlexGap: boolean; //если direction === "row" то позволит
};
```
