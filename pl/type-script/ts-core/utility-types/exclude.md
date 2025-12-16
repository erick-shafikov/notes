```ts
interface PaymentPersistent {
  id: number;
  sum: number;
  from: string;
  to: string;
}
//исключить всех, кто относится к типу, указному в generic в конце
type ExcludeEx = Exclude<"from" | "to" | Payment, string>;
/* type ExcludeEx = {
    sum: number;
    from: string;
    to: string;
} */
```

# реализация

```ts
type Exclude<T, U> = T extends U ? never : T;
```
