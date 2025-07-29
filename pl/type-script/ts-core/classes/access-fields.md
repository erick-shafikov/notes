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
