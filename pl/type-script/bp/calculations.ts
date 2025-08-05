type AsString<N extends number> = `${N}`;
type AsNumber<S extends string> = S extends `${infer N extends number}` ? N : 0;
type Parsed<S extends string> = S extends `${infer H}${infer T}`
  ? [H, T]
  : never;

type Reverse<S extends string> = S extends `${infer Head}${infer Tail}`
  ? `${Reverse<Tail>}${Head}`
  : "";

type Tuple<L extends string> = TupleOfLength<L>;

type SumReverseDigits<
  A extends string,
  B extends string,
  C extends string
> = Reverse<
  AsString<[...Tuple<A>, ...Tuple<B>, ...Tuple<C>]["length"] & number>
>;

type SumReversedDigits = <A extends string, B extends string, C extends string> = Reverse<AsString<[...Tuple<A>, ...Tuple<B>, ...Tuple<C>]>>

type SumReverseStrings<
  A extends string,
  B extends string,
  TCarry extends string = "",
  TAcc extends string = ""
> = A | B extends ""
  ? `${TAcc}${TCarry}`
  : A extends string
  ? SumReversedDigits<"0", B, TCarry, TAcc>
  : B extends string
  ? SumReversedDigits<"0", B, TCarry, TAcc>
  : [A, B] extends [
      `${infer AHead}${infer ATail}`,
      `${infer BHead}${infer BTail}`
    ]
  ? SumReverseDigits<
      AHead,
      BHead,
      TCarry
    > extends `${infer R}${infer TCarryNext}`
    ? SumReverseStrings<ATail, BTail, TCarryNext, `${TAcc}${R}`>
    : never
  : never;
