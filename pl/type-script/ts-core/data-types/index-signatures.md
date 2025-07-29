# Index signatures

```ts
interface StringArray {
  [index: number]: string; //такая сигнатура позволит проверить на наличие любое поле в объекте, который мы создадим таким образом (*)
}
const myArray: StringArray = getStringArray();
//(*)
const phones: {
  [k: string]: { country: string; area: string; number: string };
} = {
  home: { country: "x", area: "x", number: "x" },
  work: { country: "x", area: "x", number: "x" },
  fax: { country: "x", area: "x", number: "x" },
  mobile: { country: "x", area: "x", number: "x" },
};
phones["xxx"]; //нет ошибки

const secondItem = myArray[1];

interface NumberDictionary {
  [index: string]: number;
  length: number; // ok
  name: string; //свойство может быть только номером
  // Property 'name' of type 'string' is not assignable to 'string' index type 'number'.
}
interface NumberOrStringDictionary {
  [index: string]: number | string;
  length: number; // ok, length is a number
  name: string; // ok, name is a string
}
```
