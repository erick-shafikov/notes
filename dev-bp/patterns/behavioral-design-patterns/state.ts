/* 
Шаблон позволяет менять поведение класса при изменении состояния.
Шаблон «Состояние» реализует машину состояний объектно ориентированным способом. Это достигается с помощью:
реализации каждого состояния в виде производного класса интерфейса шаблона «Состояние»,
реализации переходов состояний (state transitions) посредством вызова методов, определённых вышестоящим классом (superclass).
Шаблон «Состояние» — это в некотором плане шаблон «Стратегия», при котором возможно переключение 
текущей стратегии с помощью вызова методов, 
определённых в интерфейсе шаблона.


State Позволяет объекту изменять свое поведение когда изменяется его внутреннее состояние. 
Context.Add(ObjA); ObjA.ChangeState(“A"); ObjA.ChangeState(“B”);
*/

interface WritingState {
  write: (words: string) => void;
}

class UpperCase implements WritingState {
  public write(words: string) {
    console.log(words.toUpperCase());
  }
}

class LowerCase implements WritingState {
  public write(words: string) {
    console.log(words.toLowerCase());
  }
}

class Default implements WritingState {
  public write(words: string) {
    console.log(words);
  }
}

class TextEditor {
  protected state: WritingState;

  constructor(state: WritingState) {
    this.state = state;
  }

  public setState(state: WritingState) {
    this.state = state;
  }

  public type(words: string) {
    this.state.write(words);
  }
}

const editor = new TextEditor(new Default());

editor.type("First line");

editor.setState(new UpperCase());

editor.type("Second line");
editor.type("Third line");

editor.setState(new LowerCase());

editor.type("Fourth line");
editor.type("Fifth line");

// Output:
// First line
// SECOND LINE
// THIRD LINE
// fourth line
// fifth line
