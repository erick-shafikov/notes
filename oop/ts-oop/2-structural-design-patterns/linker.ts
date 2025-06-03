/* КОМПОНОВЩИК
    Шаблон «Компоновщик» позволяет клиентам обрабатывать отдельные объекты в едином порядке.
    Шаблон «Компоновщик» описывает общий порядок обработки группы объектов, словно это одиночный экземпляр объекта. 
    Суть шаблона — компонование объектов в древовидную структуру для представления иерархии от частного к целому. 
    Шаблон позволяет клиентам одинаково обращаться к отдельным объектам и к группам объектов.

    объединяет в одну структуру разные экземпляры классов, храня их в массиве, добавляя функционал 
    позволяющий узнавать сводную информацию
*/
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
    protected roles: any[];

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
    roles: any[];

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
    protected employees: Employee[];

    public addEmployee(employee: Employee) {
        this.employees.push(employee);
    }

    public getNetSalaries(): number {
        let netSalary = 0;

        this.employees.forEach(
            (employee) => (netSalary += employee.getSalary())
        );

        return netSalary;
    }
}
// создаем структуру
const john = new Developer('John Doe', 12000);
const jane = new Designer('Jane Doe', 15000);

// Включение их в штат
const organization = new Organization();
organization.addEmployee(john);
organization.addEmployee(jane);

// вычисляем их ЗП
console.log('Net salaries: ', organization.getNetSalaries());

// более мелкие классы хранятся внутри общего класса, в котором есть доступ к любому другому

export {};
