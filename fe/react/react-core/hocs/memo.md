Второй аргумент функция сравнения пропсов компонента

```tsx
const memoizedComponent = React.memo(
  originalComponent,
  (prevProps, nextProps) => {
    return (
      prevProps.todoItem.important != nextProps.todoItem.important ||
      prevProps.todoItem.itemText != nextProps.todoItem.itemText
    );
  }
);
```

Пример

```tsx
const ToDo = (props) => {
  return (
    <ErrorBoundary errorUI={<ToDoErrorBoundary {...props} />}>
      <Inner {...props} />
    </ErrorBoundary>
  );
};

export default React.memo(ToDo, (prevProps, nextProps) => {
  return !(
    prevProps.todoItem.completed != nextProps.todoItem.completed ||
    prevProps.todoItem.important != nextProps.todoItem.important ||
    prevProps.idUpdating === prevProps.todoItem.id ||
    nextProps.idUpdating === nextProps.todoItem.id
  );
});
```

Какой-то пример

```tsx
// Здесь мемоизация не работает
<MemoizedComponent>
  <div />
<MemoizedComponent>

// И здесь мемоизация тоже не работает
<MemoizedComponent>
  <MyOtherComponent />
<MemoizedComponent>

// мемоизация сработает с примитивами
<MemoizedComponent>
  строка
<MemoizedComponent>
<MemoizedComponent>
  {123}
<MemoizedComponent>
<MemoizedComponent>
  {true}
<MemoizedComponent>

// или надо мемоизировать дочерний компонент
const memoChild = useMemo(() => <div />, []);
<MemoizedComponent>
  {memoChild}
<MemoizedComponent>

const memoChild = useMemo(() => <MyOtherComponent />, []);
<MemoizedComponent>
  {memoChild}
<MemoizedComponent>
```
