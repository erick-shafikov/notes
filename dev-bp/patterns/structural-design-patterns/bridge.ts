export {};
/* МОСТ
Шаблон «Мост» — это предпочтение компоновки наследованию. 
Подробности реализации передаются из одной иерархии другому объекту с отдельной иерархией.
Шаблон «Мост» означает отделение абстракции от реализации, чтобы их обе можно было изменять 
независимо друг от друга.

Отделяет абстракцию от ее реализации, так что они могут изменяться независимо. 
var obj = new ConcreteA(); obj.DoSomething(); obj = new ConcreteB(); obj.DoSomething();
*/
interface WebPage {
  theme: Theme;
  getContent: VoidFunction;
}

class About implements WebPage {
  theme: Theme;
  constructor(theme: Theme) {
    this.theme = theme;
  }

  public getContent() {
    return "About page in " + this.theme.getColor();
  }
}

class Careers implements WebPage {
  theme: Theme;

  constructor(theme: Theme) {
    this.theme = theme;
  }

  public getContent() {
    return "Careers page in " + this.theme.getColor();
  }
}

//интерфейс темы
interface Theme {
  getColor: VoidFunction;
}

class DarkTheme implements Theme {
  public getColor() {
    return "Dark Black";
  }
}
class LightTheme implements Theme {
  public getColor() {
    return "Off white";
  }
}
class AquaTheme implements Theme {
  public getColor() {
    return "Light blue";
  }
}

const darkTheme = new DarkTheme();

const about = new About(darkTheme);
const careers = new Careers(darkTheme);

console.log(about.getContent()); // "About page in Dark Black";
console.log(careers.getContent()); // "Careers page in Dark Black";
