# ПОСЕТИТЕЛЬ

Шаблон «Посетитель» позволяет добавлять будущие операции для объектов без их модифицирования. Шаблон — это способ отделения алгоритма от структуры объекта, в которой он оперирует. Результат отделения — возможность добавлять новые операции в существующие структуры объектов без их модифицирования. Это один из способов соблюдения принципа открытости/закрытости (open/closed principle).

Visitor (Посетитель) - Добавляет метод для работы со структурой объектов. `List.Add(Student("A")); List.Add(Student("B")); List.Visit(new VoteVisitor());`

```ts
interface Animal {
  accept: (operation: AnimalOperation) => void;
}

// Посетитель
interface AnimalOperation {
  visitMonkey: (monkey: Monkey) => void;
  visitLion: (lion: Lion) => void;
  visitDolphin: (dolphin: Dolphin) => void;
}

class Monkey implements Animal {
  public shout() {
    console.log('Ooh oo aa aa!');
  }

  public accept(operation: AnimalOperation) {
    operation.visitMonkey(this);
  }
}

class Lion implements Animal {
  public roar() {
    console.log('Roaaar!');
  }

  public accept(operation: AnimalOperation) {
    operation.visitLion(this);
  }
}

class Dolphin implements Animal {
  public speak() {
    console.log('Tuut tuttu tuutt!');
  }

  public accept(operation: AnimalOperation) {
    operation.visitDolphin(this);
  }
}

class Speak implements AnimalOperation {
  public visitMonkey(monkey: Monkey) {
    monkey.shout();
  }

  public visitLion(lion: Lion) {
    lion.roar();
  }

  public visitDolphin(dolphin: Dolphin) {
    dolphin.speak();
  }
}

class Jump implements AnimalOperation {
  public visitMonkey(monkey: Monkey) {
    console.log('Jumped 20 feet high! on to the tree!');
  }

  public visitLion(lion: Lion) {
    console.log('Jumped 7 feet! Back on the ground!');
  }

  public visitDolphin(dolphin: Dolphin) {
    console.log('Walked on water a little and disappeared');
  }
}

const monkey = new Monkey();
const lion = new Lion();
const dolphin = new Dolphin();

const speak = new Speak();
const jump = new Jump();

monkey.accept(speak); // Ooh oo aa aa!
monkey.accept(jump);  // Jumped 20 feet high! on to the tree!

lion.accept(speak);   // Roaaar!
lion.accept(jump);    // Jumped 7 feet! Back on the ground!

dolphin.accept(speak); // Tuut tuttu tuutt!
dolphin.accept(jump);  // Walked on water a little and disappeared
```
