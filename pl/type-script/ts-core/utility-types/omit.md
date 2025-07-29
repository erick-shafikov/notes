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
