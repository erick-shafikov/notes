```ts
interface PaymentPersistent {
  id: number;
  sum: number;
  from: string;
  to: string;
}
// Забираем только тот тип, который указан в конце
type ExtractEx = Extract<"from" | "to" | Payment, string>;
/* type ExtractEx = "from" | "to" */
```

# реализация

```ts
type Extract<T, U> = T extends U ? T : never;
```
