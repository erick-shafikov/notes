# interfaces и types

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

# наследование

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

# рекурсивные типы

```ts
type NestedNumbers = number | NestedNumbers[];
const val: NestedNumbers = [1, [3, [4, 5]], 7, 9];

// Пример с json-объектом
type JSONPrimitive = string | number | boolean | null;
type JSONValue = JSONPrimitive | JSONObject | JSONArray;
type JSONObject = { [key: string]: JSONValue };
type JSONArray = JSONValue[];
```

# Inference with Template Literals

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
