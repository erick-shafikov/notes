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
