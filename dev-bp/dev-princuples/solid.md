# SOLID

SOLID — пять принципов объектно-ориентированного проектирования, которые делают код понятным, гибким и удобным для расширения.

---

## S — Single Responsibility Principle (Принцип единственной ответственности)

**Один класс = одна причина для изменения.**

Если класс делает несколько разных вещей — он «знает слишком много». При изменении одной из этих вещей вы рискуете сломать другие. Признак нарушения: класс меняется по разным поводам (добавили новое поле в БД → правим этот класс; поменяли формат отчёта → снова правим его).

```ts
// ❌ Нарушение: класс делает всё сразу
class UserManager {
  getUser(id: string) { /* запрос к БД */ }
  formatUserReport(user: User) { /* форматирование */ }
  sendEmail(user: User, text: string) { /* отправка письма */ }
}

// ✅ Разделяем ответственности
class UserRepository {
  getUser(id: string) { /* запрос к БД */ }
}

class UserReportFormatter {
  format(user: User) { /* форматирование отчёта */ }
}

class EmailService {
  send(to: string, text: string) { /* отправка письма */ }
}
```

> Теперь изменение логики email не затрагивает работу с БД.

---

## O — Open/Closed Principle (Принцип открытости-закрытости)

**Класс открыт для расширения, но закрыт для изменения.**

Когда нужно добавить новое поведение — не трогай старый код, а добавляй новый. Иначе каждое новое требование будет ломать существующую логику.

```ts
// ❌ Нарушение: при добавлении нового типа скидки надо лезть в метод
class Discount {
  calculate(type: string, price: number) {
    if (type === 'vip') return price * 0.8;
    if (type === 'promo') return price * 0.9;
    // каждый новый тип — новый if внутри
    return price;
  }
}

// ✅ Расширяем через полиморфизм — старый код не трогаем
interface DiscountStrategy {
  calculate(price: number): number;
}

class VipDiscount implements DiscountStrategy {
  calculate(price: number) { return price * 0.8; }
}

class PromoDiscount implements DiscountStrategy {
  calculate(price: number) { return price * 0.9; }
}

// Новый тип скидки — просто новый класс, старое не трогаем
class SeasonalDiscount implements DiscountStrategy {
  calculate(price: number) { return price * 0.7; }
}
```

---

## L — Liskov Substitution Principle (Принцип подстановки Лисков)

**Подкласс должен быть полностью заменяем своим базовым классом.**

Если в коде написано `Shape`, туда можно подставить `Circle` или `Rectangle`, и всё будет работать как ожидается. Если подкласс ломает это ожидание (бросает исключение или ведёт себя иначе) — принцип нарушен.

```ts
// ❌ Нарушение: Square не может честно реализовать setWidth/setHeight
class Rectangle {
  width = 0;
  height = 0;
  setWidth(w: number)  { this.width = w; }
  setHeight(h: number) { this.height = h; }
  area() { return this.width * this.height; }
}

class Square extends Rectangle {
  setWidth(w: number)  { this.width = w;  this.height = w; } // ломает контракт
  setHeight(h: number) { this.width = h;  this.height = h; }
}

function resize(r: Rectangle) {
  r.setWidth(5);
  r.setHeight(2);
  // ожидаем area = 10, но Square вернёт 4 — сюрприз!
}

// ✅ Выносим общий контракт в интерфейс без конфликтующих сеттеров
interface Shape {
  area(): number;
}

class Rect implements Shape {
  constructor(private w: number, private h: number) {}
  area() { return this.w * this.h; }
}

class Sq implements Shape {
  constructor(private side: number) {}
  area() { return this.side ** 2; }
}
```

---

## I — Interface Segregation Principle (Принцип разделения интерфейсов)

**Не заставляй классы реализовывать методы, которые им не нужны.**

Большой «жирный» интерфейс — источник проблем: классы вынуждены реализовывать ненужные методы (часто заглушками), а любое изменение интерфейса затрагивает всех его наследников.

```ts
// ❌ Нарушение: принтер без WiFi вынужден реализовывать sendWifi
interface Printer {
  print(): void;
  scan(): void;
  sendWifi(): void;
}

class SimplePrinter implements Printer {
  print() { /* ok */ }
  scan() { throw new Error('Not supported'); }      // заглушка
  sendWifi() { throw new Error('Not supported'); }  // заглушка
}

// ✅ Разбиваем на узкие интерфейсы
interface Printable { print(): void; }
interface Scannable { scan(): void; }
interface WifiSender { sendWifi(): void; }

class BasicPrinter implements Printable {
  print() { /* только то, что умеет */ }
}

class AllInOnePrinter implements Printable, Scannable, WifiSender {
  print() {}
  scan() {}
  sendWifi() {}
}
```

---

## D — Dependency Inversion Principle (Принцип инверсии зависимостей)

**Зависеть нужно от абстракций, а не от конкретных реализаций.**

Высокоуровневый модуль не должен знать, что именно стоит за интерфейсом. Это позволяет менять реализации без изменения потребителя (например, сменить MySQL на PostgreSQL, или реальный email-сервис на заглушку в тестах).

```ts
// ❌ Нарушение: OrderService жёстко привязан к конкретному классу
class MySQLDatabase {
  save(order: Order) { /* сохранение в MySQL */ }
}

class OrderService {
  private db = new MySQLDatabase(); // жёсткая зависимость!
  placeOrder(order: Order) { this.db.save(order); }
}

// ✅ Зависим от интерфейса — конкретику передаём снаружи (DI)
interface Database {
  save(order: Order): void;
}

class MySQLDatabase implements Database {
  save(order: Order) { /* MySQL */ }
}

class InMemoryDatabase implements Database {
  save(order: Order) { /* для тестов */ }
}

class OrderService {
  constructor(private db: Database) {} // зависит от абстракции
  placeOrder(order: Order) { this.db.save(order); }
}

// Использование
const service = new OrderService(new MySQLDatabase());
const testService = new OrderService(new InMemoryDatabase()); // в тестах
```

---

## Пример, объединяющий все пять принципов

Система обработки заказов в интернет-магазине.

```ts
// === Интерфейсы (I, D) ===
interface OrderRepository {
  save(order: Order): void;
}

interface PaymentProcessor {
  charge(amount: number, cardToken: string): boolean;
}

interface NotificationSender {
  notify(email: string, message: string): void;
}

// === Конкретные реализации (O — добавляем не трогая потребителей) ===
class PostgresOrderRepository implements OrderRepository {
  save(order: Order) { /* SQL INSERT */ }
}

class StripePaymentProcessor implements PaymentProcessor {
  charge(amount: number, cardToken: string) { /* вызов Stripe API */ return true; }
}

class EmailNotificationSender implements NotificationSender {
  notify(email: string, message: string) { /* отправка письма */ }
}

// Легко добавить альтернативы, не трогая OrderService:
class SmsNotificationSender implements NotificationSender {
  notify(email: string, message: string) { /* отправка SMS */ }
}

// === Сущность (S — только данные заказа) ===
class Order {
  constructor(
    public id: string,
    public amount: number,
    public customerEmail: string,
    public cardToken: string,
  ) {}
}

// === Главный сервис (S, D, L) ===
// OrderService отвечает только за координацию шагов оформления заказа.
// Реализации БД, платёжки и уведомлений передаются снаружи.
class OrderService {
  constructor(
    private repo: OrderRepository,        // D: зависит от абстракции
    private payment: PaymentProcessor,    // D
    private notifier: NotificationSender, // D
  ) {}

  placeOrder(order: Order): void {
    const paid = this.payment.charge(order.amount, order.cardToken);

    if (!paid) throw new Error('Payment failed');

    this.repo.save(order);                              // S: не знает, как сохранять
    this.notifier.notify(order.customerEmail, 'Order confirmed!'); // S: не знает, как слать
  }
}

// === Сборка (можно менять любую деталь независимо) ===
const orderService = new OrderService(
  new PostgresOrderRepository(),  // завтра можно заменить на MongoOrderRepository
  new StripePaymentProcessor(),   // завтра — на PayPalPaymentProcessor
  new EmailNotificationSender(),  // завтра — на SmsNotificationSender
);
```

**Как здесь проявляется каждый принцип:**

| Принцип | Где |
|---------|-----|
| **S** | `OrderService` координирует, не хранит и не шлёт — у каждого класса одна роль |
| **O** | Новый способ оплаты = новый класс `implements PaymentProcessor`, старый код не трогаем |
| **L** | Любой `NotificationSender` взаимозаменяем — `OrderService` не заметит подмены |
| **I** | Три узких интерфейса вместо одного «бога» с методами `save/charge/notify` |
| **D** | `OrderService` зависит от `OrderRepository`, а не от `PostgresOrderRepository` напрямую |
