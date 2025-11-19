# inheritance

```ts
class User {
  private _firstName: string = "";
  private _secondName: string = "";

  set firstName(name: string) {
    this._firstName = name;
  }

  set secondName(name: string) {
    this._secondName = name;
  }

  get fullName() {
    return this._firstName + this._secondName;
  }
}

class Employee extends User {
  constructor(public jobTitle: string) {
    super();
  }

  work() {
    //console.log(this._firstName); // Ошибка нет доступа к private
  }
}
```

Инициализация наследованных конструкторов

```ts
class Coord {
  long: number;
  lat!: number;

  message = "1"; //(*)

  constructor(lat: number, long: number) {
    this.lat = lat; //инициализация поля
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

  constructor(lat: number, long: number, name: string) {
    super(lat, long);
  } //обязательная операция при наследовании, при инициализации в log будет выводится 1
}

let m = new MapLocation(1, 1, "sdf");
```

Порядок инициализации:

- инициализируются поля базового класса
- запускается конструктор базового класса
- инициализируются поля наследованного класса
- запускается конструкторы унаследованных классов

- при наследовании в случае переписывании методов для разных аргументов задавать их опционально
