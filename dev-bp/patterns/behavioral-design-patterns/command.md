# КОМАНДА

Шаблон «Команда» позволяет инкапсулировать действия в объекты.
Ключевая идея — предоставить средства отделения клиента от получателя.
В шаблоне «Команда» объект используется для инкапсуляции всей информации,
необходимой для выполнения действия либо для его инициирования позднее.
Информация включает в себя имя метода; объект, владеющий методом; значения параметров метода.

Command - (Команда) - Инкапсулирует запрос в виде объекта с поддержкой различных команд.
Command.DoSomething(); Command.Redo(); Command.Undo();

```ts
//экземпляр с функционалом
class Bulb {
  public turnOn() {
    console.log("Bulb has been lit");
  }

  public turnOff() {
    console.log("Darkness!");
  }
}

interface Command {
  execute: VoidFunction;
  undo: VoidFunction;
  redo: VoidFunction;
}

//Функционал объекта
class TurnOn implements Command {
  protected bulb: Bulb;

  constructor(bulb: Bulb) {
    this.bulb = bulb;
  }

  public execute() {
    this.bulb.turnOn();
  }

  public undo() {
    this.bulb.turnOff();
  }

  public redo() {
    this.execute();
  }
}

class TurnOff implements Command {
  protected bulb: Bulb;

  constructor(bulb: Bulb) {
    this.bulb = bulb;
  }

  public execute() {
    this.bulb.turnOff();
  }

  public undo() {
    this.bulb.turnOn();
  }

  public redo() {
    this.execute();
  }
}

// класс активации главного метода
class RemoteControl {
  public submit(command: Command) {
    command.execute();
  }
}
//создаем экземпляр
const bulb = new Bulb();

//создаем функционал
const turnOn = new TurnOn(bulb);
const turnOff = new TurnOff(bulb);

//создаем исполнитель
const remote = new RemoteControl();

//передаем в исполнитель команду, которая выполняет действия с экземпляром
remote.submit(turnOn); // Лампочка зажглась!
remote.submit(turnOff); // Темнота!
```
