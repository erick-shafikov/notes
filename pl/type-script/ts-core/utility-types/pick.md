```ts
interface PaymentPersistent {
  id: number;
  sum: number;
  from: string;
  to: string;
}
//взять что-то из типов , взять только 'from' и 'to'
type PaymentRequisite = Pick<PaymentPersistent, "from" | "to">;
/* type PaymentRequisite = {
    from: string;
    to: string;
} */
```
