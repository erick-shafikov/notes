Шаблон «Цепочка ответственности» позволяет создавать цепочки объектов.
Запрос входит с одного конца цепочки и движется от объекта к объекту, пока не будет найден подходящий обработчик.
Шаблон «Цепочка ответственности» содержит исходный управляющий объект и ряд обрабатывающих объектов.
Каждый обрабатывающий объект содержит логику, определяющую типы командных объектов,
которые он может обрабатывать, а остальные передаются по цепочке следующему обрабатывающему объекту.

- Chain of Response (Цепочка обязанностей) - Избегает связи отправителя с приемником, давая объекту возможность для обработки запроса. Employee.SetSupervisor(Manager); Manager.SetSupervisor(Director); Employee.Execute();

```ts
abstract class Account {
  protected successor: Account;
  protected balance: number;

  //конструктор принимает экземпляр класса для перенаправления
  public setNext(account: Account) {
    this.successor = account;
  }
  //при вызове pay в зависимости от баланса будет либо перенаправление либо проброс исключения
  public pay(amountToPay: number) {
    if (this.canPay(amountToPay)) {
      console.log("Paid %s using %s", amountToPay);
    } else if (this.successor) {
      console.log("Cannot pay using %s. Proceeding ..");
      this.successor.pay(amountToPay);
    } else {
      throw new Error("None of the accounts have enough balance");
    }
  }

  public canPay(amount: number): boolean {
    return this.balance >= amount;
  }
}

class Bank extends Account {
  protected balance: number;

  constructor(balance: number) {
    super();
    this.balance = balance;
  }
}

class Paypal extends Account {
  protected balance: number;

  constructor(balance: number) {
    super();
    this.balance = balance;
  }
}

class Bitcoin extends Account {
  protected balance: number;

  constructor(balance: number) {
    super();
    this.balance = balance;
  }
}

//инициализация
const bank = new Bank(100); // У банка баланс 100
const paypal = new Paypal(200); // У Paypal баланс 200
const bitcoin = new Bitcoin(300); // У Bitcoin баланс 300

//создание цепочек
bank.setNext(paypal);
paypal.setNext(bitcoin);

// Начнём с банка
bank.pay(259);

// Выходной вид
// ==============
// Нельзя оплатить с помощью банка. Обрабатываю...
// Нельзя оплатить с помощью Paypal. Обрабатываю...
// Оплачено 259 с помощью Bitcoin!
```
