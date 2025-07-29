# Mapped types

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

# readonly

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
