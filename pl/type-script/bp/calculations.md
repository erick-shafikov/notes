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

Вычисления

```ts
// массив длиной N
type TupleOfLength<
  N extends number,
  Acc extends unknown[] = []
> = Acc["length"] extends N ? Acc : TupleOfLength<N, [...Acc, unknown]>;
// превратить число в строку
type AsString<N extends number> = `${N}`;
// превратить строку в число !!! N extends number - превратит в число
type AsNumber<S extends string> = S extends `${infer N extends number}` ? N : 0;
// посимвольная обработка строк, отдаст H как первый символ, в T передаст хвост
type Parsed<S extends string> = S extends `${infer H}${infer T}`
  ? [H, T]
  : never;
// отщипываем первый символ, отправляем в конец
type Reverse<S extends string> = S extends `${infer Head}${infer Tail}`
  ? `${Reverse<Tail>}${Head}`
  : "";

type Tuple<L extends string> = TupleOfLength<AsNumber<L>>;

type SumReversedDigits<
  A extends string,
  B extends string,
  C extends string
> = Reverse<
  AsString<[...Tuple<A>, ...Tuple<B>, ...Tuple<C>]["length"] & number>
>;

type SumReversedStrings<
  A extends string,
  B extends string,
  TCarry extends string = "",
  TAcc extends string = ""
> = A | B extends ""
  ? `${TAcc}${TCarry}` //проверка на кончились ли цифры
  : A extends ""
  ? SumReversedStrings<"0", B, TCarry, TAcc> //если разное количество цифр A > B
  : B extends ""
  ? SumReversedStrings<A, "0", TCarry, TAcc> //если разное количество цифр A < B
  : [A, B] extends [
      `${infer AHead}${infer ATail}`,
      `${infer BHead}${infer BTail}`
    ]
  ? SumReversedDigits<
      AHead,
      BHead,
      TCarry
    > extends `${infer R}${infer TCarryNext}`
    ? SumReversedStrings<ATail, BTail, TCarryNext, `${TAcc}${R}`>
    : never
  : never;

type Sum<A extends number, B extends number> = AsNumber<
  Reverse<SumReversedStrings<Reverse<AsString<A>>, Reverse<AsString<B>>>>
>;

type S = Sum<2, 3>;

// умножение

type DecrementMap = {
  "1": "0";
  "2": "1";
  "3": "2";
  "4": "3";
  "5": "4";
  "6": "5";
  "7": "6";
  "8": "7";
  "9": "8";
};

type DecrementReversed<A> = A extends `${infer AH}${infer AT}`
  ? AH extends "0"
    ? `9${DecrementReversed<AT>}`
    : `${DecrementMap[AH & keyof DecrementMap]}${AT}`
  : never;

type ProductReversedStrings<
  A extends string,
  B extends string,
  R extends string = "0"
> = A extends "0"
  ? R
  : B extends "0"
  ? R
  : A extends `${infer AH}${infer AT}`
  ? AH extends "0"
    ? ProductReversedStrings<AT, `0${B}, R`>
    : ProductReversedStrings<DecrementReversed<A>, B, SumReversedStrings<R, B>>
  : R;

type Product<A extends number, B extends number> = AsNumber<
  Reverse<ProductReversedStrings<Reverse<AsString<A>>, Reverse<AsString<B>>>>
>;

type P = Product<2, 3>;

// вычисление выражений

type Evaluated<TExpression extends string> =
  TExpression extends `${infer L}(${infer TNested})${infer R}`
    ? Evaluated<`${L}${Evaluated<TNested>}${R}`>
    : TExpression extends `${infer A} + ${infer B}`
    ? Sum<Evaluated<A>, Evaluated<B>>
    : TExpression extends `${infer A} * ${infer B}`
    ? Product<Evaluated<A>, Evaluated<B>>
    : AsNumber<TExpression>;

type E = Evaluated<"(2 + 3) * 5">;
```

добавляем в eval

```ts
declare global {
  function eval<E extends string>(expr: E): Evaluated<E>;
}
```
