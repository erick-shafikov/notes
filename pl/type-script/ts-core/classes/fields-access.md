# типы и доступность полей

private, public - есть только в ts

```ts
class Vehicle {
  public make: string; //поле по умолчанию

  private damages: string[]; //невозможно обратиться к полю из экземпляра класса, только внутри, так и наследники не могут обратиться
  private _model: string; //_ работает только в TS
  #price: number; // сокращенная запись private

  protected run: number; //недоступно извне, доступно только через методы, сеттеры и геттеры, при переопределении в наследниках – становится публичным для экземпляров наследников

  readonly name: string = "world"; //поле только для чтения, может сочетаться с другими асессорами private, public

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

vh = new Vehicle();
// vh.run //ошибка так как run protected

class EuroTruck extends Vehicle {
  setRun(km: number) {
    // нет ошибки так как protected
    this.run = km / 0.62; //this.damage - error
    // console(_model) //ошибка
  }
}
```
