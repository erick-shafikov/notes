# constructor

```ts
class Coord {
  long: number; //поле, по умолчанию требует инициализацию в конструкторе
  lat!: number; //поле, которое не требует инициализации

  message = "message";

  constructor(lat: number, long: number) {
    this.lat = lat; //инициализация поля
    this.long = long;
  }
}
```

# шорткат для полей

позволяется обойтись без

```ts
class User {
  constructor(public name: string) {} // может быть и private
}

// вместо

class User {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}
```
