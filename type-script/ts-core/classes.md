# Классы

```ts
class Coord {
  message = "1"; //(*)
  lat!: number; //поле, которое не требует инициализации
  long: number;
  protected test() {
    if (this.lat > 0) {
    }
  }
  private test2() {}

  computeDistance(newLat: number, newLong: number) {
    this.test();
  }

  constructor(lat: number, long: number) {
    this.lat = lat; //инициализация класса
    this.long = long;
    console.log(this.message); //так как это конструктор запуститься раньше конструктора mapLocation, то в консоль всегда будет выводится 1(*)}
  }
}
//tsconfig - strictPropertyInitialization
const point = new Coord(0, 1); //экземпляр класса

class MapLocation extends Coord {
  //наследование класса
  message = "2"; //(*)
  private _name: string;

  get name() {
    return this._name;
  }

  set name(s: string) {
    this._name = s;
  }

  override computeDistance(newLat: number, newLong: number) {
    this.test();
  }

  constructor(lat: number, long: number, name: string) {
    super(lat, long);
  } //обязательная операция при наследовании, при инициализации в log будет выводится 1
}

let m = new MapLocation(1, 1, "sdf");
```

- геттеры и сеттеры не могут быть асинхронными
- при наследовании в случае переписывании методов для разных аргументов задавать их опционально

Порядок инициализации:

- инициализируются поля базового класса
- запускается конструктор базового класса
- инициализируются поля наследованного класса
- запускается конструкторы унаследованных классов

<!-- типы и доступность полей -------------------------------------------------------------------------------------------------------------->

# типы и доступность полей

```ts
class Vehicle {
  public make: string; //поле по умолчанию

  private damages: string[]; //невозможно обратить из экземпляра класса, только внутри, так и наследники не могут обратиться
  private _model: string; //работает только в TS
  #price: number; // сокращенная запись private

  protected run: number; //недоступно извне, доступно только через методы и сеттеры и геттеры, при переопределении в наследниках – становится публичным для экземпляров наследников

  readonly name: string = "world"; //поле только для чтения

  set model(m: string) {
    //метод обращения через геттер
    this._model = m;
    this.#price = 100; //Доступность #полей
  }

  get model() {
    return this._model;
  }

  isPriceEqual(v: Vehicle) {
    this.#price === v.#price; //проверка эквивалентных приватных свойств
  }

  private addDamage(damage: string) {
    this.damages.push(damage);
  }
}

class EuroTruck extends Vehicle {
  setRun(km: number) {
    this.run = km / 0.62; //this.damage - error
  }
}
```

# типы и доступность методов

```ts
class Coord {
  message = "1";
  lat!: number;
  long: number;

  protected test() {
    if (this.lat > 0) {
      //доступен только внутри класса (**),
    }
  }
  private test2() {
    //доступен внутри класса Coord и недоступен в других
  }

  computeDistance(newLat: number, newLong: number) {
    this.test(); //доступен в классе Coord
  }

  constructor(lat: number, long: number) {
    this.lat = lat;
    this.long = long;
    console.log(this.message);
  }
} //демонстрация взаимодействия с private свойствами класса

const point = new Coord(0, 1); //экземпляр класса
point.test(); //Property 'test' is protected and only accessible within class 'Coord' and its subclasses.ts(2445)(**)

class MapLocation extends Coord {
  message = "2"; //(*)
  private _name: string;
  get name() {
    return this._name;
  }
  set name(s: string) {
    this._name = s;
  }

  override computeDistance(newLat: number, newLong: number) {
    this.test();
    //(**) метод test доступен и в наследуемом объекте а tets2() – нет Property 'test2' is private and only accessible within class 'Coord'

    this.test2();
  }

  constructor(lat: number, long: number, name: string) {
    super(lat, long);
  }
}

let m = new MapLocation(1, 1, "sdf");
m.test(); //Property 'test' is protected and only accessible within class 'Coord' and its subclasses.ts(2445)(**)
```

## Перегрузка методов

```ts
class User {
  skills: string[];
  addSkill(skill: string): void; //в зависимости от типа аргумента позволяет реализовать разный функционал
  addSkill(skill: string[]): void;
  addSkill(skill: string | string[]): void {
    if (typeof skill === "string") {
      this.skills.push(skill);
    } else {
      this.skills.concat(skill);
    }
  }
} //перегрузка функций
function run(distance: string): string;
function run(distance: number): number;
function run(distance: number | string): number | string {
  if (typeof distance === "number") {
    return 1;
  } else {
    return "";
  }
}
```

## static

Статический метод – метод, не имеющий доступа к состоянию (полям) объекта, то есть к переменной this. Слово «статический» используется в том смысле, что статические методы не относятся к динамике объекта, не используют и не меняют его состояния.

```ts
class UserService {
  static dbStaticOnLy: any;
  private static db: any;

  static getUser(id: number) {
    //недоступен из
    return UserService.db.findByTd(id); //здесь можно использовать асинхронные операции
  }
  static {
    //код для инициализации статистических полей, недоступны асинхронные методы
  }
}
UserService.dbStaticOnLy; //доступен из класса
```

Если задать static constructor() { } , тогда экземпляр класса будет недоступен для создания, можно будет только пользоваться его статичными полями

## override

```ts
class Coord {
  message = "1";
  lat!: number;
  long: number;
  protected test() {
    if (this.lat > 0) {
    }
  }
  private test2() {}
  computeDistance(newLat: number, newLong: number) {
    this.test();
  }
  constructor(lat: number, long: number) {
    this.lat = lat;
    this.long = long;
    console.log(this.message);
  }
}
const point = new Coord(0, 1); //экземпляр класса

class MapLocation extends Coord {
  message = "2"; //(*)
  private _name: string;

  get name() {
    return this._name;
  }
  set name(s: string) {
    this._name = s;
  }
  override computeDistance(newLat: number, newLong: number) {
    //при удаление или изменения этого метода в родителе возможна потеря
  }

  constructor(lat: number, long: number, name: string) {
    super(lat, long);
  }
}

let m = new MapLocation(1, 1, "sdf");
```

<!-- this ---------------------------------------------------------------------------------------------------------------------------------->

# this

this теряется при использовании методов в других объектах и наследовании

```ts
class PaymentLoseContext {
  private data: Date = new Date();
  getDate() {
    return this.data;
  }
}
const newPayment1 = new PaymentLoseContext();

const user1 = {
  id: 1,
  paymentData: newPayment1.getDate(), //потеря контекста
};

class PaymentWithContext {
  private data: Date = new Date();
  getDate(this: PaymentWithContext) {
    //привязка контекста через ключевое this
    return this.data;
  }
  getDateArrow = () => {
    return this.data; //второй вариант привязка с помощью стрелочной функции
  };
}

const newPayment2 = new PaymentWithContext();

const user2 = {
  id: 1,
  paymentData: newPayment2.getDate(), //
  paymentData2: newPayment2.getDate.bind(newPayment2), //привязываем с помощью Bind
};

class brokeArrowLogic extends PaymentWithContext {
  save() {
    return super.getDateArrow(); //Не будет работать
    return this.getDateArrow(); //будет работать
  }
}
```

<!-- абстрактные классы -------------------------------------------------------------------------------------------------------------------->

# абстрактные классы

Абстрактные классы можно только наследовать, но использовать как экземпляр класса – невозможно

```ts
abstract class Base {
  //пример абстрактного класса в отличие от типов - это класс
  print(s: string) {
    console.log(s);
  }
  abstract error(a: string): void; //абстрактный метод абстрактного класса, заставит реализовать этот метод
}
new Base(); //невозможно создать экземпляр этого класса
class BaseExtended extends Base {
  //а наследовать - можно
  error(a: string): void {}
}
new BaseExtended().print("s");
```

## абстракции и implements

Имплементация - взаимодействие интерфейса и класса связать не через два класса, а через адаптер

```ts
interface LoggerService {
  //абстракция как должен работать класс, там должен быть метод log,который ничего не возвращает
  log: (s: string) => void;
}
class Logger implements LoggerService {
  public log(s: string) {
    //без указания типа, по умолчанию тип s будет any, сам метод публичный по умолчанию - public
    console.log(s);
  }
  private error() {} //приватный метод, который не даст доступ l.error(); - Property 'error' is private and only accessible within class 'Logger'.ts(2341)
  private a = ""; //пример приватного
}
const l = new Logger();
l.log("d");
```

<!-- Generics ------------------------------------------------------------------------------------------------------------------------------>

# Классы и generic

```ts
// пример 1
class MyClass {
  static a = "1"; //ненужно инициализировать
  static {} //статический блок инициализации
}
MyClass.a = "asd";
class AnotherClass<T> {
  a: T;
}
const b = new AnotherClass<string>(); //b.a будет строкой

// пример 2
class Resp<D, E> {
  //класс из двух полей
  data?: D;
  error?: E;
  constructor(data?: D, error?: E) {
    if (data) {
      this.data = data;
      this.error = error;
    }
  }
}
const res = new Resp<string, number>("data", 123);
class HTTPResp<F> extends Resp<string, number> {
  //расширяем с жоп generic
  code?: F;
  setCode(code: F) {
    this.code = code;
  }
}
const res2 = new HTTPResp();
```

## Миксины

```ts
type Constructor = new (...args: any[]) => {}; //для любого тип конструктора
type GConstructor<T = {}> = new (...args: any[]) => T; //ограничим с помощью generic, получает T и возвращаетT
class List {
  constructor(public items: string[]) {}
}
class Accordion {
  isOpened?: boolean;
}
type ListType = GConstructor<List>; //конструктор тип
type AccordionType = GConstructor<Accordion>;
//класс расширяет лист с доп функционалом в виде получения первого элемента (стандартное расширение)
class ExtendedCLass extends List {
  first() {
    return this.items[0]; //Доп функционал
  }
}
//MIXIN функция которая сливает 2 класса, в функцию передаём класс
function ExtendedList<TBase extends ListType & AccordionType>(Base: TBase) {
  return class ExtendedList extends Base {
    first() {
      return this.items[0]; //Доп. функционал
    }
  };
}
class AccordionList {
  //для слива двух классов
  isOpened?: boolean;
  constructor(public items: string[]) {}
}
const list = ExtendedList(AccordionList);
const res = new list(["1", "2", "3"]);
```
