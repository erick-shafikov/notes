# BigInt

- числа больше 2^53 - 1
- Number.MAX_SAFE_INTEGER - самое большое число в JS
- нельзя использовать в Math
- typeof BigInt === "bigint":

```js
const theBiggestInt = 9007199254740991n;

const alsoHuge = BigInt(9007199254740991); // 9007199254740991n
const hugeString = BigInt("9007199254740991"); // 9007199254740991n
const hugeHex = BigInt("0x1fffffffffffff"); //9007199254740991n
const hugeBin = BigInt(
  "0b11111111111111111111111111111111111111111111111111111"
); //9007199254740991n

0n === 0; //  false
```

# Методы

BigInt.asIntN() - статический метод, который позволяет перенести BigInt-значение в целое число со знаком
BigInt.asIntN()
