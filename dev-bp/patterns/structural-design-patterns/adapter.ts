/* АДАПТЕР
    Шаблон «Адаптер» позволяет помещать несовместимый объект в обёртку, чтобы он оказался совместимым с другим классом.
    Шаблон проектирования «Адаптер» позволяет использовать интерфейс существующего класса как другой интерфейс. 
    Этот шаблон часто применяется для обеспечения работы одних классов с другими без изменения их исходного кода.

    Т.е. Класс, который принимает в конструктор объект и адаптирует его к классу, от которого унаследовался

    Преобразует интерфейс класса в другой интерфейс, который клиенты ожидают. 
    Target obj = new Adapter(); 
    obj.DoSomething();
*/

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
export {};
