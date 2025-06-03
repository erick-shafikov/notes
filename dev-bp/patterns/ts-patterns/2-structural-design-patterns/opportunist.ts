/* Приспособленец
Шаблон применяется для минимизирования использования памяти или вычислительной стоимости 
за счёт общего использования как можно большего количества одинаковых объектов.
«Приспособленец» — это объект, минимизирующий использование памяти за счёт общего с другими, 
такими же объектами использования как можно большего объёма данных. Это способ применения многочисленных объектов, 
когда простое повторяющееся представление приведёт к неприемлемому потреблению памяти.
*/

class KarakTea {}

// Действует как фабрика и экономит чай
class TeaMaker {
    protected availableTea: Record<string, any> = {};

    public make(preference: string) {
        if (!this.availableTea[preference]) {
            this.availableTea[preference] = new KarakTea();
        }

        return this.availableTea[preference];
    }
}

class TeaShop {
    protected orders: Record<number, any>;
    protected teaMaker: TeaMaker;

    public constructor(teaMaker: TeaMaker) {
        this.teaMaker = teaMaker;
    }

    public takeOrder(teaType: string, table: number) {
        this.orders[table] = this.teaMaker.make(teaType);
    }

    public serve() {
        for (let tea in this.orders) {
            console.log('Serving tea to table# ');
        }
    }
}

const teaMaker = new TeaMaker();
const shop = new TeaShop(teaMaker);

shop.takeOrder('less sugar', 1);
shop.takeOrder('more milk', 2);
shop.takeOrder('without sugar', 5);

shop.serve();
// Serving tea to table# 1
// Serving tea to table# 2
// Serving tea to table# 5

export {};
