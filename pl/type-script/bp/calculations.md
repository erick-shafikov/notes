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

# пример с New Function

```ts
/**
 * Calculates the square of an orthogonal parallelepiped
 **/
const square = new Function(
  "a",
  "b",
  "c",
  `
  const s1 = a * b;
  const s2 = b * c;
  const s3 = a * c;

  return (s1 + s2 + s3) * 2;
`
);

// const s: 178
const s = square(11, 4, 3);

console.log(s); // 178

// TYPE COERCION
type AsString<N extends number> = `${N}`;

type AsNumber<T extends string | number> =
  `${T}` extends `${infer N extends number}` ? N : 0;

// STRING HELPERS
type Reverse<A extends string> = A extends `${infer AH}${infer AT}`
  ? `${Reverse<AT>}${AH}`
  : "";

type Split<
  TString extends string,
  TSeparator extends string
> = TString extends `${infer L}${TSeparator}${infer R}`
  ? [L, ...Split<R, TSeparator>]
  : [TString];

type Trim<
  TString extends string,
  TOccurrence extends string = " " | "\n"
> = TString extends
  | `${TOccurrence}${infer TTrimmed}`
  | `${infer TTrimmed}${TOccurrence}`
  ? Trim<TTrimmed>
  : TString;

// TUPLE HELPERS
type Zip<A extends unknown[], B extends unknown[]> = [A, B] extends [
  [infer AHead, ...infer ATail],
  [infer BHead, ...infer BTail]
]
  ? [[AHead, BHead], ...Zip<ATail, BTail>]
  : [];

type FromEntries<T extends [PropertyKey, unknown][]> = {
  [P in T[number] as P[0]]: P[1];
};

// CONTEXT
type Context = Record<string, number>;

type WithVar<
  TContext extends Context,
  TName extends string,
  TValue extends number
> = Omit<TContext, TName> & { [key in TName]: TValue };

type VarValue<TContext extends Context, TName extends string> = TContext[TName &
  keyof TContext];

type WithReturn<TContext extends Context, TValue extends number> = WithVar<
  TContext,
  "@return",
  TValue
>;

type ReturnedValue<TContext extends Context> = VarValue<TContext, "@return">;

// MATH
type TupleOfLength<
  N extends number,
  Acc extends unknown[] = []
> = Acc["length"] extends N ? Acc : TupleOfLength<N, [...Acc, unknown]>;

type SumReversedDigits<
  A extends string,
  B extends string,
  C extends string = "0"
> = Reverse<
  AsString<
    [
      ...TupleOfLength<AsNumber<A>>,
      ...TupleOfLength<AsNumber<B>>,
      ...TupleOfLength<AsNumber<C>>
    ]["length"] &
      number
  >
>;

type SumReversedStrings<
  A extends string,
  B extends string,
  TCarry extends string = "",
  TAcc extends string = ""
> = A | B extends ""
  ? `${TAcc}${TCarry}`
  : A extends ""
  ? SumReversedStrings<"0", B, TCarry, TAcc>
  : B extends ""
  ? SumReversedStrings<A, "0", TCarry, TAcc>
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

// https://github.com/type-challenges/type-challenges/issues/5814

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
    ? ProductReversedStrings<AT, `0${B}`, R>
    : ProductReversedStrings<DecrementReversed<A>, B, SumReversedStrings<R, B>>
  : R;

type Product<A extends number, B extends number> = AsNumber<
  Reverse<ProductReversedStrings<Reverse<AsString<A>>, Reverse<AsString<B>>>>
>;

// EVALUATION
type Evaluated<
  TExpression extends string,
  TContext extends Context = {}
> = TExpression extends `${infer L}(${infer TNested})${infer R}`
  ? Evaluated<`${L}${Evaluated<TNested, TContext>}${R}`, TContext>
  : TExpression extends `${infer A} + ${infer B}`
  ? Sum<Evaluated<A, TContext>, Evaluated<B, TContext>>
  : TExpression extends `${infer A} * ${infer B}`
  ? Product<Evaluated<A, TContext>, Evaluated<B, TContext>>
  : VarValue<TContext, TExpression> extends never
  ? AsNumber<TExpression>
  : VarValue<TContext, TExpression>;

// INTERPRETATION
type Interpreted<
  TInstruction extends string,
  TContext extends Context
> = TInstruction extends `const ${infer TVariable} = ${infer TExpression}`
  ? WithVar<TContext, TVariable, Evaluated<TExpression, TContext>>
  : TInstruction extends `return ${infer TExpression}`
  ? WithReturn<TContext, Evaluated<TExpression, TContext>>
  : TContext;

type InterpretedInstructions<
  TInstructions extends string[],
  TContext extends Context
> = TInstructions extends [
  infer TInstruction extends string,
  ...infer TRest extends string[]
]
  ? InterpretedInstructions<TRest, Interpreted<TInstruction, TContext>>
  : TContext;

// CODE PASRING
type TrimLines<TLines extends string[]> = TLines extends [
  infer TFirst extends string,
  ...infer TRest extends string[]
]
  ? [Trim<TFirst>, ...TrimLines<TRest>]
  : [];

type ParsedInstructions<TBody extends string> = TrimLines<
  Split<Trim<TBody, ";">, ";">
>;

// FUNCTION PARSING
type ConstructedFunction<TBody extends string, TArgNames extends string[]> = (<
  TArgValues extends number[]
>(
  ...args: TArgValues
) => FunctionResult<TBody, TArgNames, TArgValues>) &
  ((...args: unknown[]) => unknown);

type FunctionResult<
  TBody extends string,
  TArgNames extends string[],
  TArgValues extends number[]
> = ReturnedValue<
  InterpretedInstructions<
    ParsedInstructions<TBody>,
    FunctionInitContext<TArgNames, TArgValues>
  >
>;

type FunctionInitContext<
  TArgNames extends string[],
  TArgValues extends number[]
> = FromEntries<Zip<TArgNames, TArgValues> & [string, number][]>;

// GLOBAL OVERLOADS
declare global {
  interface FunctionConstructor {
    new <TArgNames extends string[], TBody extends string>(
      ...params: [...TArgNames, TBody]
    ): ConstructedFunction<TBody, TArgNames>;
  }

  function eval<TExpression extends string>(
    expression: TExpression
  ): Evaluated<TExpression>;
}

// Necessary for global types augmentation
export { s, result };
```
