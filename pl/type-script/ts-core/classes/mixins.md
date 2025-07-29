# Миксины

```ts
type Constructor = new (...args: any[]) => {}; //для любого тип конструктора
type GConstructor<T = {}> = new (...args: any[]) => T; //ограничим с помощью generic, получает T и возвращаетT

class List {
  constructor(public items: string[]) {}
}

class Accordion {
  isOpened?: boolean;
}

type ListType = GConstructor<List>; //конструктор тип
type AccordionType = GConstructor<Accordion>;

//класс расширяет лист с доп функционалом в виде получения первого элемента (стандартное расширение)
class ExtendedCLass extends List {
  first() {
    return this.items[0]; //Доп функционал
  }
}

//MIXIN функция которая сливает 2 класса, в функцию передаём класс
function ExtendedList<TBase extends ListType & AccordionType>(Base: TBase) {
  return class ExtendedList extends Base {
    first() {
      return this.items[0]; //Доп. функционал
    }
  };
}

class AccordionList {
  //для слива двух классов
  isOpened?: boolean;
  constructor(public items: string[]) {}
}

const list = ExtendedList(AccordionList);
const res = new list(["1", "2", "3"]);
```
