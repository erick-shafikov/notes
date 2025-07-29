# Required

Создает тип, состоящий из всех свойств Type, для которых установлено значение required. Противоположность Partial, все поля интерфейса должны быть обязательными

```ts
interface Props {
  a?: number;
  b?: string;
}
const obj: Props = { a: 5 };
const obj2: Required<Props> = { a: 5 };
// Property 'b' is missing in type '{ a: number; }' but required in type 'Required<Props>'.
```
