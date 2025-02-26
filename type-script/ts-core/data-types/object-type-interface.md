## interfaces и types

использования типов для аргументов функции

```ts
type coord = { lat: number; long: number }; //типы могут быть объектом
type ID = number | string; //тип может быть union’ом, интерфейс - нет
interface ICoord {
  //интерфейс могут быть только объектом
  lat: number;
  long: number;
}

function compute(coord: ID) {} //используем наш новый тип вместо coord: {lat: number, long: number}
```

наследование через interfaces

```ts
interface AnimalInterface {
  name: string;
}
interface DogInterface extends AnimalInterface {
  //принцип наследования
  tail?: boolean; //необязательное свойство tail
}
const dogByInterface: DogInterface = {
  name: "someString",
};
```

наследование через types

```ts
type AnimalType = {
  name: string;
};

type DogType = AnimalType & {
  //благодаря & DogType будет содержать свойства name и tail
  tail: boolean;
};

const dogByType: DogType = {
  name: "",
  tail: true,
};
```

Объединение типов по мере выполнения кода

можно по ходу выполнения дополнять интерфейсы

```ts
interface Dog {
  name: string;
}
interface Dog {
  //в ходе выполнения можно добавить в интерфейс еще свойство
  tail: boolean;
}

const dog: Dog = {
  name: "asd",
  tail: true,
};
```

# Разница type и interface

- Types не могут участвовать в слиянии объектов
- interface более удобен при наследовании
- interfaces могут определять только объекты, а не примитивы
- Всегда используйте interface если не нужна какая-то особенность types
- Тип может быть юнионом, а интерфейс может быть только объектом.
- Тип не может быть классом. Интерфейс может быть унаследован.
- Types расширяются с помощью амперсанда.
- Два интерфейса с одинаковыми именами будут объединены в один, а два одинаковых имени для types вызовут ошибку

# Оператор & с объектами и строками

для объектов – создает объединение, для примитивов ищет пересечение

```ts
type TA = { a1: "a1"; a2: "a2" };
type TB = { b1: "b1"; b2: "b2" };
type TAandTB = TA & TB;

const objTAandTB: TAandTB = {
  //все поля обязательные, нельзя добавить новое или пропустить
  a1: "a1",
  a2: "a2",
  b1: "b1",
  b2: "b2",
};

// оператор & для строковых
type TC = "c1" | "c2";
type TD = "d1" | "d2";
type TE = "c1" | "d2";
type TCandTD = TC & TD; //TCandTD === never так как нет общего
type TCandTCD = TC & TE; //c1 - Общее между 'c1' | 'c2' и 'c1' | 'd2‘
```

# Оператор | с объектами и строками

для объектов – пересечение или объединение, для строк - объединение

```ts
// оператор | для объектных литералов
type TA = { a1: "a1"; a2: "a2" };
type TB = { b1: "b1"; b2: "b2" };
type TAandTB = TA | TB;

const objTAandTB1: TAandTB = {
  //обязательные поля одного из типов
  a1: "a1",
  a2: "a2",
};

const objTAandTB2: TAandTB = {
  //без проблем добавляет поля второго типа
  b1: "b1",
  b2: "b2",
  a1: "a1",
};

// оператор & для строковых
type TC = "c1" | "c2";
type TD = "d1" | "d2";
type TE = "c1" | "d2";
type TCandTD = TC | TD;
const TCandTD = "c1"; //один из типов c1, c2, d1, d2
type TCandTCD = TC | TE; //"c1" | "c2" | "d2" - объединил все типы
```

## рекурсивные типы

```ts
type NestedNumbers = number | NestedNumbers[];
const val: NestedNumbers = [1, [3, [4, 5]], 7, 9];

// Пример с json-объектом
type JSONPrimitive = string | number | boolean | null;
type JSONValue = JSONPrimitive | JSONObject | JSONArray;
type JSONObject = { [key: string]: JSONValue };
type JSONArray = JSONValue[];
```

## Index signatures

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

## Indexed Access Types

```ts
interface Role {
  name: string;
}

interface Permission {
  endDate: Date;
}

interface User {
  name: string;
  roles: Role[]; //string[]
  permission: Permission; // {endDate: Date}
}

const user: User = {
  name: "Vasya",
  roles: [],
  permission: {
    endDate: new Date(),
  },
};

// ----------------------------------------------------------------------
const nameUser = user["name"]; //string
let rolesName: "roles" = "roles";
type rolesType = User["roles"]; //type rolesType = Role[]
type rolesType2 = User[typeof rolesName]; //rolesName === 'roles' => rolesType2 === Roles[]
type roleType = User["roles"][number]; //получить элемент массива, ключ === number
type roleType2 = User["permission"]["endDate"]; //получить элемент массива, вытаскиваем из массива тип roleType2 = Date
const roles = ["admin", "user", "super-user"] as const; //теперь это tuple с тремя элементами
type roleTypes = (typeof roles)[number]; //type roleTypes 'admin' | 'user' | 'super-user'

// ----------------------------------------------------------------------
type Person = { age: number; name: string; alive: boolean };
type I1 = Person["age" | "name"]; //string | number определение аллиаса ключей
type AliveOrName = "alive" | "name"; //или так
type I5 = Person[AliveOrName]; //string | boolean

type I3 = Person[keyof Person]; //string | number | boolean использование с keyof
```

## Mapped types

```ts
type OptionsFlags<Type> = {
  [Property in keyof Type]: boolean; // тип достающий из дженерика все ключи со значением boolean
};

type Concrete<Type> = {
  -readonly [Property in keyof Type]-?: Type[Property]; // для обязательных свойств и readonly свойств
};

type ModifierToAccess1<T> = {
  [Property in keyof T]+?: boolean; //все будут необязательные
};

//модифицирование свойств
type Getters<Type> = {
  //string & Property так как Property может быть number, сужаем тип
  [Property in keyof Type as `get${Capitalize<
    string & Property
  >}`]: () => Type[Property];
};

type ModifierToAccess5<T> = {
  //изменять названия
  +readonly [Property in keyof T as `canAccess${string & Property}`]-?: boolean; //всем ключам добавим canAccess
};

interface Person {
  name: string;
  age: number;
  location: string;
}
type LazyPerson = Getters<Person>; //type LazyPerson = { getName: () => string; getAge: () => number; getLocation: () => string }
```

исключение свойств из объекта с помощью утилиты Exclude

```ts
type RemoveKindField<Type> = {
  [Property in keyof Type as Exclude<Property, "kind">]: Type[Property];
};

interface Circle {
  kind: "circle";
  radius: number;
}

type KindlessCircle = RemoveKindField<Circle>; //type KindlessCircle = { radius: number }

type ModifierToAccess<T> = {
  //исключит 'canAccessAdminPanel' с помощью Exclude
  +readonly [Property in keyof T as Exclude<
    `canAccess${string & Property}`,
    "canAccessAdminPanel"
  >]-?: boolean;
};
```

## Inference with Template Literals

```ts
type PropEventSource<Type> = {
  //принимает объект
  on(
    eventName: `${string & keyof Type}Changed`, //достаем из keyof
    callback: (newValue: any) => void
  ): void; //добавляет метод on, который вызывается с ключом + Changed
};

declare function makeWatchedObject<Type>( //функция которая переделывает в объект с методом on
  obj: Type
): Type & PropEventSource<Type>; //второй вариант

type PropEventSource2<Type> = {
  on<Key extends string & keyof Type>( //используем дженерик, который достанет ключ
    eventName: `${Key}Changed`,
    callback: (newValue: Type[Key]) => void //тогда ключ будет доступен и в коллбеке
  ): void;
};

const person = makeWatchedObject({
  firstName: "Person1",
  lastName: "Person2",
  age: 26,
});

person.on("firstNameChanged", () => {});
```

## массивы

```ts
const numTriplet: [number, number, number] = [7, 7, 7];
numTriplet.length; //3
numTriplet.pop();
numTriplet.pop();
numTriplet.pop();
numTriplet.length; //3 ts не видит, что мы вытащили все элементы из, лечится с помощью readonly
```

```ts
type Array2<T> = [T, ...T[]]; //массив с одним обязательным элементом
type Array2<T> = [T, T, ...T[]]; //массив с двумя обязательными элементом
```

## readonly

```ts
interface SomeType {
  readonly prop: string;
}
 
function doSomething(obj: SomeType) {
  //OK
  console.log(`prop has the value '${obj.prop}'.`);
 
  //Cannot assign to 'prop' because it is a read-only property.
  obj.prop = "hello";
}


interface SomeType {
  readonly prop!: string; //необязательный параметр
}

```
