/* ОДИНОЧКА
    Шаблон позволяет удостовериться, что создаваемый объект — единственный в своём классе.
    Шаблон «Одиночка» позволяет ограничивать создание класса единственным объектом. 
    Это удобно, когда для координации действий в рамках системы требуется, чтобы объект был единственным в своём классе.
*/

class President {
    self: President | null;

    private constructor() {
        // Прячем конструктор
    }

    public static getInstance(): President | null {
        let self: null | President = null;
        if (!self) {
            self = new President();
        }

        return self;
    }

    private clone() {
        // Отключаем клонирование
    }

    private wakeUp() {
        // Отключаем десериализацию
    }
}

const president1 = President.getInstance();
const president2 = President.getInstance();

console.log(president1 === president2); // true
