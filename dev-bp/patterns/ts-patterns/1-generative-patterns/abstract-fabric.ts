/* АБСТРАКТНАЯ ФАБРИКА
Это фабрика фабрик. То есть фабрика, группирующая индивидуальные, но взаимосвязанные/взаимозависимые фабрики 
без указания для них конкретных классов.
Шаблон «Абстрактная фабрика» описывает способ инкапсулирования группы индивидуальных фабрик, 
объединённых некой темой, без указания для них конкретных классов.
*/
// разные типы дверей
interface Door {
    getDescription: VoidFunction;
}

class WoodenDoor implements Door {
    public getDescription() {
        console.log('I am a wooden door');
    }
}

class IronDoor implements Door {
    public getDescription() {
        console.log('I am an iron door');
    }
}
// разные типы специалистов
interface DoorFittingExpert {
    getDescription: VoidFunction;
}

class Welder implements DoorFittingExpert {
    public getDescription() {
        console.log('I can only fit iron doors');
    }
}

class Carpenter implements DoorFittingExpert {
    public getDescription() {
        console.log('I can only fit wooden doors');
    }
}

interface DoorFactory {
    makeDoor: () => Door;
    makeFittingExpert: () => DoorFittingExpert;
}

// Фабрика деревянных дверей возвращает плотника и деревянную дверь
class WoodenDoorFactory implements DoorFactory {
    public makeDoor(): Door {
        return new WoodenDoor();
    }

    public makeFittingExpert(): DoorFittingExpert {
        return new Carpenter();
    }
}

// Фабрика стальных дверей возвращает стальную дверь и сварщика
class IronDoorFactory implements DoorFactory {
    public makeDoor(): Door {
        return new IronDoor();
    }

    public makeFittingExpert(): DoorFittingExpert {
        return new Welder();
    }
}
// использование
const woodenFactory = new WoodenDoorFactory();

const door = woodenFactory.makeDoor();
const expert = woodenFactory.makeFittingExpert();

door.getDescription(); // Output: Я деревянная дверь
expert.getDescription(); // Output: Я могу устанавливать только деревянные двери

// Same for Iron Factory
const ironFactory = new IronDoorFactory();

const door1 = ironFactory.makeDoor();
const expert2 = ironFactory.makeFittingExpert();

door1.getDescription(); // Output: Я стальная дверь
expert2.getDescription();

export {};
