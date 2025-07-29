# Distributive conditional types

```ts
type ToArray<Type> = Type extends any ? Type[] : never;
// в первом случае все зависит от первого присвоения
type StrArrOrNumArr1 = ToArray<string | number>; //type StrArrOrNumArr1 = string[] | number[]
const strArrOrNumArr1: StrArrOrNumArr1 = ["s"];
const strArrOrNumArr2: StrArrOrNumArr1 = [1];
const strArrOrNumArr3: StrArrOrNumArr1 = ["s", 1]; //ошибка

// при Distributive  разбивается
type ToArrayNonDist<Type> = [Type] extends [any] ? Type[] : never;
// 'StrArrOrNumArr' is no longer a union.
type StrArrOrNumArr = ToArrayNonDist<string | number>; //type StrArrOrNumArr = (string | number)[]
const strArrOrNumArr4: StrArrOrNumArr = ["s", 1]; //ошибки нет

type TToArray<T> = T[]; //тоже самое
```
