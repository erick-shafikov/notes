/* СТРОИТЕЛЬ
Шаблон позволяет создавать разные свойства объекта, избегая загрязнения конструктора (constructor pollution). 
Это полезно, когда у объекта может быть несколько свойств. 
Или когда создание объекта состоит из большого количества этапов.
Шаблон «Строитель» предназначен для поиска решения проблемы антипаттерна Telescoping constructor.
*/

type TBurgerParams = {
    size: number;
    cheese: boolean;
    pepperoni: boolean;
    lettuce: boolean;
    tomato: boolean;
};

class Burger {
    protected size: number;

    protected cheese = false;
    protected pepperoni = false;
    protected lettuce = false;
    protected tomato = false;

    constructor(builder: TBurgerParams) {
        this.size = builder.size;
        this.cheese = builder.cheese;
        this.pepperoni = builder.pepperoni;
        this.lettuce = builder.lettuce;
        this.tomato = builder.tomato;
    }
}
//класс для создания параметров конструктора
class BurgerBuilder {
    public size;

    public cheese = false;
    public pepperoni = false;
    public lettuce = false;
    public tomato = false;

    constructor(size: number) {
        this.size = size;
    }

    public addPepperoni() {
        this.pepperoni = true;
        return this;
    }

    public addLettuce() {
        this.lettuce = true;
        return this;
    }

    public addCheese() {
        this.cheese = true;
        return this;
    }

    public addTomato() {
        this.tomato = true;
        return this;
    }

    public build(): Burger {
        return new Burger(this);
    }
}
// использование
const burger = new BurgerBuilder(14)
    .addPepperoni()
    .addLettuce()
    .addTomato()
    .build();
