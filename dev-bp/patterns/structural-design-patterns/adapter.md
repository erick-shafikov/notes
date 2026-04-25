# АДАПТЕР

Адаптер — это структурный паттерн, который позволяет объектам с несовместимыми интерфейсами работать вместе. Он оборачивает объект (Adaptee) и приводит его интерфейс к ожидаемому интерфейсу (Target).

Когда использовать: есть существующий класс, который нельзя менять его интерфейс не совпадает с тем, что ожидает клиент

Структура:

Target — ожидаемый интерфейс
Adaptee — существующий класс
Adapter — реализует Target и использует Adaptee

```ts
interface Lion {
  roar: VoidFunction;
}

class AfricanLion implements Lion {
  public roar() {}
}

class AsianLion implements Lion {
  public roar() {}
}

class Hunter {
  public hunt(lion: Lion) {}
}

class WildDog {
  public bark() {}
}

//адаптер, который является экземпляром Lion, но превращает другой тип класса в Lion
class WildDogAdapter implements Lion {
  protected dog: WildDog;

  constructor(dog: WildDog) {
    this.dog = dog;
  }

  public roar() {
    this.dog.bark();
  }
}

const wildDog = new WildDog();
const wildDogAdapter = new WildDogAdapter(wildDog);

const hunter = new Hunter();
hunter.hunt(wildDogAdapter);
```
