Предотвращение рендеринга с помощью useSearch

```ts
// компонент не поменяется если `bar` поменяется
const foo = Route.useSearch({ select: ({ foo }) => foo });
```

```ts
//каждый раз ре-рендер так как возвращается компонент
const result = Route.useSearch({
  select: (search) => {
    return {
      foo: search.foo,
      hello: `hello ${search.foo}`,
    };
  },
  // изменить поведение
  structuralSharing: true,
});
```
