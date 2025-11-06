# Соединение двух объектов

```ts
type Prettify<T> = T extends object ? { [Key in keyof T]: Pretty<T[Key]> } : T;

type Data = Prettify<ShortData & AdditionalData>;
```

# ключи в объекте

```ts
type Person = { firstName: string; age: number };

function logPerson(person: Person) {
  console.log(person);
}

logPerson({ firstName: "trash", age: 21, extraProp: 21 }); // ошибка

const person = {
  firstName: "trash",
  age: 21,
  extraProp: 21,
};

logPerson(person); // нет ошибки

const person: Person = {
  firstName: "trash",
  age: 21,
  extraProp: 21, // Object literal may only specify known properties, and 'extraProp' does not exist in type 'Person'.ts(2353)
};

const person: Person = {
  firstName: "trash",
  age: 21,
  ...{ extraProp: 21 }, // ошибки нет
};
```

# {}, object, Object

```ts
//object
let foo: object;

foo = { hello: 0 }; //ok
foo = []; //ok
foo = false; //error
foo = null; //error
foo = undefined; //error
foo = 42; //error
foo = "bar"; //error
//{} - любое значение но не undefined или null
let foo: {};

foo = { hello: 0 }; //ok
foo = []; //ok
foo = false; //ok
foo = null; //error
foo = undefined; //error
foo = 42; //ok
foo = "bar"; //ok
```
