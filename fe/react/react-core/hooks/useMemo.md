# Хук useMemo()

чистые функции, которые не нужно вычислять при каждом рендере повторно

При первоначальном рендеринге useMemoвозвращает результат вызова calculateValueбез аргументов.

- нужно для вычислений. которые заметно замедляют выполнение кода
- при передачи в качестве пропса
- значение является зависимостью
- если нужно передать компонент в качестве дочернего

```tsx
function TodoList({ todos, tab, theme }) {
  const visibleTodos = useMemo(() => filterTodos(todos, tab), [todos, tab]);
  const children = useMemo(() => <List items={visibleTodos} />, [visibleTodos]);
  return <div className={theme}>{children}</div>;
}
```

```jsx
function complexCompute(num) {
  //функция дает задержку
  let i = 0;
  while (i < 1000000000) {
    i++;
  }
  return num * 2;
}
function UseMemoHook() {
  const [number, setNumber] = useState(42);
  const [colored, setColored] = useState(false); //функция, которая не имеет отношения к тяжелым вычислениям
  const styles = {
    color: colored ? "darkred" : "black",
  };
  const computed = useMemo(() => {
    //оборачиваем вызов дорогостоящей функции в useMemo
    return complexCompute(number);
  }, [number]);
  return (
    <>
      <h1 _style={styles}>calculated value:{computed}</h1>
      <button onClick={() => setNumber((prev) => prev + 1)}>increase</button>
      <button onClick={() => setNumber((prev) => prev - 1)}>decrease</button>
      <button
        className="btn btn-warning"
        onClick={() => setColored((prev) => !prev)}
      >
        Change
      </button>
    </>
  );
}
```

Пример бессмысленной мемоизации, компонент создаваемый внутри тела компонента (всегда новая ссылка)

```tsx
function Dropdown({ allItems, text }) {
  const searchOptions = { matchMode: 'whole-word', text };

  const visibleItems = useMemo(() => {
    return searchItems(allItems, searchOptions);
  }, [allItems, searchOptions]); // 🚩 Caution: Dependency on an object created in the component body
  // ...

// улучшение
  function Dropdown({ allItems, text }) {
  const searchOptions = useMemo(() => {
    return { matchMode: 'whole-word', text };
  }, [text]); // ✅ Only changes when text changes

  const visibleItems = useMemo(() => {
    return searchItems(allItems, searchOptions);
  }, [allItems, searchOptions]); // ✅ Only changes when allItems or searchOptions changes

// лучший вариант
  function Dropdown({ allItems, text }) {
  const visibleItems = useMemo(() => {
    const searchOptions = { matchMode: 'whole-word', text };
    return searchItems(allItems, searchOptions);
  }, [allItems, text]); // ✅ Only changes when allItems or text changes
```

## Проблема ссылочного типа

```jsx
function complexCompute(num) {
  //функция дает задержку
  let i = 0;
  while (i < 1000000000) {
    i++;
  }
  return num * 2;
}
function UseMemoHook() {
  const [number, setNumber] = useState(42);
  const [colored, setColored] = useState(false);
  const styles = useMemo(
    () => ({
      color: colored ? "darkred" : "black", //решаем проблему лишнего лога и создания нового объекта
    }),
    [colored]
  );
  const computed = useMemo(() => {
    return complexCompute(number);
  }, [number]);
  useEffect(() => {
    console.log("styles changed"); //лишние раз будет вызываться при каждом рендере, так ссылочный тип объекта создает новый, объекты в зависимости не совпадают, поэтому будет вызываться лог
  }, [styles]);
  return (
    <>
      <h1 _style={styles}>calculated value:{computed}</h1>{" "}
      <button onClick={() => setNumber((prev) => prev + 1)}>increase</button>   
      <button onClick={() => setNumber((prev) => prev - 1)}>decrease</button>   
      <button
        className="btn btn-warning"
        onClick={() => setColored((prev) => !prev)}
      >
        Change
      </button>
    </>
  );
}
```
