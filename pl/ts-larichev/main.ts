export default {};

class Car {
    static readonly #nextSerialNumber: null; // ===  private static nextSerialNumber
    private static generateSerialNumber() {
        //статичный метод, в нем можно получить доступ к private static, а в экземпляре нельзя
        return this.#nextSerialNumber;
    }
    static {
        //static block вызывается сразу, как класс создается, то есть при инициализации в коде
        fetch('')
            .then((response) => response.json())
            .then((data) => {
                this.#nextSerialNumber = data.mostRecentInvoiceId + 1;
            });
    }

    make: string;
    model: string;
    year: number;
    serialNumberStatic = Car.generateSerialNumber(); //обращение к статичному полю
    private _serialNumber() {
        return this._serialNumber;
    }
    protected get serialNumber() {
        return this._serialNumber;
    }

    constructor(make: string, model: string, year: number) {
        this.make = make;
        this.model = model;
        this.year = year;
    }

    honk(duration: number): string {
        return `h${'o'.repeat(duration)}nk`;
    }

    getLabel() {
        return `${this.make} ${this.model} - ${this.serialNumber}`;
    }

    equals(other: any) {
        if (other && typeof other === 'object' && #serialNumber in other)
            return other.serialNumber === this._serialNumber;
    }
}

let sedan = new Car('Honda', 'Accord', 2017);
