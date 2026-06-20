# Компоновщик

Шаблон «Компоновщик» позволяет клиентам обрабатывать отдельные объекты в едином порядке. Шаблон «Компоновщик» описывает общий порядок обработки группы объектов, словно это одиночный экземпляр объекта. Суть шаблона — компонование объектов в древовидную структуру для представления иерархии от частного к целому. Шаблон позволяет клиентам одинаково обращаться к отдельным объектам и к группам объектов. Объединяет в одну структуру разные экземпляры классов, храня их в массиве, добавляя функционал позволяющий узнавать сводную информацию. Добавляет объекты в древовидные структуры для представления сложных иерархий. Позволяет работать с одиночными объектами и их группами одинаково. Он строит древовидную структуру:

- Leaf (лист) — простой объект
- Composite (контейнер) — содержит другие элементы (Leaf или Composite)

Клиент работает с ними через общий интерфейс, в котором ему без разницы на внутреннюю структуру

```ts
interface FileSystemItem {
  getSize(): number;
}

// Лист — обычный файл
class File implements FileSystemItem {
  constructor(private size: number) {}

  public getSize(): number {
    return this.size;
  }
}

// Компоновщик — папка
class Directory implements FileSystemItem {
  private children: FileSystemItem[] = [];

  public add(item: FileSystemItem) {
    this.children.push(item);
  }

  public getSize(): number {
    return this.children.reduce((total, child) => {
      return total + child.getSize();
    }, 0);
  }
}

const file1 = new File(100);
const file2 = new File(200);

const folder1 = new Directory();
folder1.add(file1);
folder1.add(file2);

const file3 = new File(300);

const root = new Directory();
root.add(folder1);
root.add(file3);

// клиенту не важно — файл это или папка
console.log(root.getSize()); // 600
```

Клиент не различает файл это (File) или папка (Directory). React по сути работает как компоновщик - любой компонент может содержать другие компоненты, и рендер происходит одинаково

```ts
//интерфейс для отдельного сотрудника
interface Employee {
  name: string;
  salary: number;
  getName: () => string;
  setSalary: (salary: number) => void;
  getSalary: () => number;
  getRoles: () => any[];
}

class Developer implements Employee {
  salary: number;
  name: string;
  protected roles: any[] = [];

  constructor(name: string, salary: number) {
    this.name = name;
    this.salary = salary;
  }

  public getName(): string {
    return this.name;
  }

  public setSalary(salary: number) {
    this.salary = salary;
  }

  public getSalary() {
    return this.salary;
  }

  public getRoles() {
    return this.roles;
  }
}

class Designer implements Employee {
  salary;
  name;
  roles: any[] = [];

  constructor(name: string, salary: number) {
    this.name = name;
    this.salary = salary;
  }

  public getName(): string {
    return this.name;
  }

  public setSalary(salary: number) {
    this.salary = salary;
  }

  public getSalary() {
    return this.salary;
  }

  public getRoles() {
    return this.roles;
  }
}
//класс организации
class Organization {
  protected employees: Employee[] = [];

  public addEmployee(employee: Employee) {
    this.employees.push(employee);
  }

  public getNetSalaries(): number {
    let netSalary = 0;

    this.employees.forEach((employee) => (netSalary += employee.getSalary()));

    return netSalary;
  }
}
// создаем структуру
const john = new Developer("John Doe", 12000);
const jane = new Designer("Jane Doe", 15000);

// Включение их в штат
const organization = new Organization();
organization.addEmployee(john);
organization.addEmployee(jane);

// вычисляем их ЗП
console.log("Net salaries: ", organization.getNetSalaries());

// более мелкие классы хранятся внутри общего класса, в котором есть доступ к любому другому
```
