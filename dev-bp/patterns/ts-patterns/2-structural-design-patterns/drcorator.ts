/* ДЕКОРАТОР
    Шаблон «Декоратор» позволяет во время выполнения динамически изменять поведение объекта, 
    обёртывая его в объект класса «декоратора».
    Шаблон «Декоратор» позволяет подключать к объекту дополнительное поведение (статически или динамически), 
    не влияя на поведение других объектов того же класса. Шаблон часто используется 
    для соблюдения принципа единственной обязанности (Single Responsibility Principle), 
    поскольку позволяет разделить функциональность между классами для решения конкретных задач.

    Т.е. создается базовый объект, потом создается объект обложка, который принимает в конструктор оборачиваемый
    объект, имея те же поля, но модифицируя методы принятого в конструктор объекта  
*/

interface Coffee {
    getCost: () => number;
    getDescription: () => string;
}
// Базовый класс кофе
class SimpleCoffee implements Coffee {
    public getCost() {
        return 10;
    }

    public getDescription() {
        return 'Simple coffee';
    }
}
//Разновидности
class MilkCoffee implements Coffee {
    protected coffee;

    //в конструктор идет экземпляр базового класса
    constructor(coffee: Coffee) {
        this.coffee = coffee;
    }

    public getCost() {
        return this.coffee.getCost() + 2;
    }

    public getDescription() {
        return this.coffee.getDescription() + ', milk';
    }
}

class WhipCoffee implements Coffee {
    protected coffee;

    constructor(coffee: Coffee) {
        this.coffee = coffee;
    }

    public getCost() {
        return this.coffee.getCost() + 5;
    }

    public getDescription() {
        return this.coffee.getDescription() + ', whip';
    }
}

class VanillaCoffee implements Coffee {
    protected coffee;

    constructor(coffee: Coffee) {
        this.coffee = coffee;
    }

    public getCost() {
        return this.coffee.getCost() + 3;
    }

    public getDescription() {
        return this.coffee.getDescription() + ', vanilla';
    }
}

const someCoffee1 = new SimpleCoffee();
console.log(someCoffee1.getCost()); // 10
console.log(someCoffee1.getDescription()); // Simple Coffee

const someCoffee2 = new MilkCoffee(someCoffee1);
console.log(someCoffee2.getCost()); // 12
console.log(someCoffee2.getDescription()); // Simple Coffee, milk

const someCoffee3 = new WhipCoffee(someCoffee1);
console.log(someCoffee3.getCost()); // 17
console.log(someCoffee3.getDescription()); // Simple Coffee, milk, whip

const someCoffee4 = new VanillaCoffee(someCoffee1);
console.log(someCoffee4.getCost()); // 20
console.log(someCoffee4.getDescription()); // Simple Coffee, milk, whip, vanilla

// каждый следующий класс оборачивает предыдущий определенной логикой
