/* ЗАМЕСТИТЕЛЬ 
С помощью шаблона «Заместитель» класс представляет функциональность другого класса.
В наиболее общей форме «Заместитель» — это класс, функционирующий как интерфейс к чему-либо. 
Это оболочка или объект-агент, вызываемый клиентом для получения доступа к другому, «настоящему» объекту. 
«Заместитель» может просто переадресовывать запросы настоящему объекту, а может предоставлять дополнительную логику: 
кеширование данных при интенсивном выполнении операций или потреблении ресурсов настоящим объектом; 
проверка предварительных условий (preconditions) до вызова выполнения операций настоящим объектом

Т.е Принимает в конструктор объект, изменяя и расширяя его логику
*/

// Дверь с методами открыть и закрыть
interface Door {
    open: VoidFunction;
    close: VoidFunction;
}

class LabDoor implements Door {
    public open() {
        console.log('Opening lab door');
    }

    public close() {
        console.log('Closing the lab door');
    }
}

// Заместитель - логика охраняемой двери
class Security {
    protected door: Door;

    constructor(door: Door) {
        this.door = door;
    }
    // метод open со своей логикой открытия
    public open(password: string) {
        if (this.authenticate(password)) {
            this.door.open();
        } else {
            console.log("Big no! It ain't possible.");
        }
    }

    public authenticate(password: string) {
        return password === 'ecr@t';
    }

    public close() {
        this.door.close();
    }
}

//создаем экземпляр заместителя и работаем с ним
const door = new Security(new LabDoor());
door.open('invalid'); // Big no! It ain't possible.

door.open('ecr@t'); // Opening lab door
door.close(); // Closing lab door

export {};
