/* Фасад
    Шаблон «Фасад» предоставляет упрощённый интерфейс для сложной подсистемы.
    «Фасад» — это объект, предоставляющий упрощённый интерфейс для более крупного тела кода, например библиотеки классов.
    
    То есть есть объект с методами, а объект фасад принимает в конструктор экземпляр объекта и объединяет 
    Разные методы в один

    Предоставляет унифицированный интерфейс для включения других интерфейсов в подсистеме. 
    Façade.MethodFromObjA(); Façade.MethodFromObjB();
*/

// есть компьютер со множеством функции
class Computer {
  public getElectricShock() {
    console.log("Ouch!");
  }

  public makeSound() {
    console.log("Beep beep!");
  }

  public showLoadingScreen() {
    console.log("Loading..");
  }

  public bam() {
    console.log("Ready to be used!");
  }

  public closeEverything() {
    console.log("Bup bup bup buzzzz!");
  }

  public sooth() {
    console.log("Zzzzz");
  }

  public pullCurrent() {
    console.log("Haaah!");
  }
}

// у компьютера есть функции включения, выключения и другие, которые объединяют некоторые ф-ции компьютера
class ComputerFacade {
  protected computer: Computer;

  constructor(computer: Computer) {
    this.computer = computer;
  }

  public turnOn() {
    this.computer.getElectricShock();
    this.computer.makeSound();
    this.computer.showLoadingScreen();
    this.computer.bam();
  }

  public turnOff() {
    this.computer.closeEverything();
    this.computer.pullCurrent();
    this.computer.sooth();
  }
}

//использование
const computer = new ComputerFacade(new Computer());
computer.turnOn(); // Ouch! Beep beep! Loading.. Ready to be used!
computer.turnOff(); // Bup bup buzzz! Haah! Zzzzz

export {};
