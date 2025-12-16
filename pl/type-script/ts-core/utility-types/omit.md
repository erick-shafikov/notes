# Omit

```ts
interface PaymentPersistent {
  id: number;
  sum: number;
  from: string;
  to: string;
}
// убрать что то из типа убрать id выше
type Payment = Omit<PaymentPersistent, "id">;
/* type Payment = {
    sum: number;
    from: string;
    to: string;
} */
```

# реализация

```ts
type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>;
```

# особенности поведения

Omit юнион-типа будет пустым объектом

```ts
type A = { a: string; b: string };
type B = { c: string; d: string };

type Result = Omit<A | B, "c">;
```

Исправленный тип

```ts
type DistributiveOmit<T, K extends keyof any> = T extends any
  ? Omit<T, K>
  : never;
```
