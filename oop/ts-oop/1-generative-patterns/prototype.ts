/* ПРОТОТИП
    Объект создаётся посредством клонирования существующего объекта.
    Шаблон «Прототип» используется, когда типы создаваемых объектов определяются экземпляром-прототипом, 
    клонированным для создания новых объектов.
*/

class Sheep {
    protected name: string;
    protected category: string;

    constructor(name: string, category = 'Mountain Sheep') {
        this.name = name;
        this.category = category;
    }

    public setName(name: string) {
        this.name = name;
    }

    public getName() {
        return this.name;
    }

    public setCategory(category: string) {
        this.category = category;
    }

    public getCategory() {
        return this.category;
    }
}

const original = new Sheep('Jolly');
console.log(original.getName()); // Джолли
console.log(original.getCategory()); // Горная овечка

// Клонируйте и модифицируйте, что нужно
const cloned = Object.assign(original, {});
cloned.setName('Dolly');
console.log(cloned.getName()); // Долли
console.log(cloned.getCategory()); // Горная овечка
