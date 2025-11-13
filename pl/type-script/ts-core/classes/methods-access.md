# типы и доступность методов

```ts
class Coord {
  message = "1";
  lat!: number;
  long: number;

  constructor(lat: number, long: number) {
    this.lat = lat;
    this.long = long;
    console.log(this.message);
  }

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

# static

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

# override

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
