# getters and setters

позволяет настроить поведение поля

```ts
class MapLocation {
  private _name: string;

  constructor(name: string) {
    this._name = name;
  }

  get name() {
    return this._name;
  }

  set name(s: string) {
    this._name = s;
  }
}

let m = new MapLocation(1, 1, "sdf");
```

- геттеры и сеттеры не могут быть асинхронными
