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
