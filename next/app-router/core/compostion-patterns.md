## composition-patterns

- вместо контекста можно воспользоваться кешированием
- использование server-only (библиотека)
- Использование сторонних библиотек через ре-экспорт в клиентских компонентах
- использовать клиентские компоненты ниже в дереве компонентов
- Серверные компоненты нельзя экспортировать в клиентские, но их можно передать пропсом

```jsx
const ContextProvider = ({ children }) => {
  const [s, ss] = useState();
  return <Context.Provider value={s}>{children}</Context.Provider>;
};
```

Приоритетность рендеринга:
static - нет динамики, закешированные данные, не смотрим в кукуи (generateStaticParams для динамики)
dynamic - если не сработал static
