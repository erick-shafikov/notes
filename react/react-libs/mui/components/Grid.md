```jsx
const GridComponent = () => <Grid />;
```

```ts
type GridProps = {
  children: ReactNode;
  classes: object;
  columns: number | object; //количество колонок
  columnSpacing: Array<number | string> | number | object | string; //переопределит columns
  component: elementType; //тег
  container: boolean; //если это контейнер, если не этот пропс то нужен компонент контейнер
  direction:
    | "column-reverse"
    | "column"
    | "row-reverse"
    | "row"
    | Array<"column-reverse" | "column" | "row-reverse" | "row">
    | object;
  item: boolean; //если это элемент контейнера
  //аналогично md, sm, xl, xs
  lg: "auto" | number | bool; // true - растянется, при числе займет столько столбцов из 12, auto - займет расстояние по ширине своей ширине
  sx: Array<func | object | bool> | func | object; // стилизация
  rowSpacing: Array<number | string> | number | object | string;
  spacing: Array<number | string> | number | object | string; //расстояния между элементами
  wrap: "nowrap" | "wrap-reverse" | "wrap"; //display: wrap
  zeroMinWidth: boolean;
};
```
