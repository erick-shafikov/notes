# satisfies

проверит общий тип и по значению определит более узкий тип

```ts
const dataEntries: Record<string | number> = {
  entry:1 1,
  entry2: 2,
};

dataEntries.entry3 //нет ошибки
```

```ts
const dataEntries = {
  entry:1 1,
  entry2: 2,
} satisfies Record<string | number>;

dataEntries.entry3 //ошибка
```
