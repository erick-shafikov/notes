Пример сложения

```ts
//пример с низкой производительностью из-за глубины рекурсии
type TupleOfLength<
  N extends number,
  Acc extends unknown[] = []
> = Acc["length"] extends N ? ACC : TupleOfLength<N, [...Acc, unknown]>;

type Sum<A extends number, S extends number> = [
  ...TupleOfLength<A>,
  ...TupleOfLength<A>
]["length"];

type S = Sum<15, 2>; //17
```

сложения в столбик

```ts
type SumReversedString<A extends string, B extends string, TCarry extends string='', TAcc extends string =''> = A | B extends "" ? `${TAcc}${TCarry}`
: A extends string ? SumReversedString<'0', B, TCarry, TAcc >
: B extends string ? SumReversedString<'0', B, TCarry, TAcc >
: [A, B] extends [`${infer AHead}${infer ATail}`, `${infer BHead}${infer BTail}`,] ? SumReverseDigits<AHead, BHead, TCarry> extends `${infer R}${infer TCarryNext}`
? SumReverseDigits<ATail, BTail, TCarryNext, `${TAcc}${R}`
: never
:never
```
