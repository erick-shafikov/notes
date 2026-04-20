# Фабричный метод

Определяет интерфейс для создания объекта, при этом позволяет решить, какие подклассы класса. Используются для создания экземпляра. FactoryA.Create(); FactoryB.Create();

```ts
// базовый класс с методом deliver
interface Transport {
  deliver(): void;
}

// конкретная реализация
class Truck implements Transport {
  deliver() {
    console.log("Deliver by land (truck)");
  }
}

class Ship implements Transport {
  deliver() {
    console.log("Deliver by sea (ship)");
  }
}

// фабричный метод, абстракция по созданию
abstract class Logistics {
  // в каждом Transport есть  createTransport
  abstract createTransport(): Transport;

  planDelivery() {
    const transport = this.createTransport();
    transport.deliver();
  }
}

class RoadLogistics extends Logistics {
  createTransport(): Transport {
    return new Truck();
  }
}

class SeaLogistics extends Logistics {
  createTransport(): Transport {
    return new Ship();
  }
}

// использование
const logistics: Logistics = new RoadLogistics();
logistics.planDelivery();
```

реализация через switch

```ts
type UserRole = "admin" | "user";

interface User {
  permissions: string[];
}

class Admin implements User {
  permissions = ["read", "write", "delete"];
}

class RegularUser implements User {
  permissions = ["read"];
}

function createUser(role: UserRole): User {
  switch (role) {
    case "admin":
      return new Admin();
    case "user":
      return new RegularUser();
    default:
      throw new Error("Unknown role");
  }
}

// использование
const user = createUser("admin");
console.log(user.permissions);
```

через словарь

```ts
type UserRole = "admin" | "user";

interface User {
  permissions: string[];
}

class Admin implements User {
  permissions = ["read", "write", "delete"];
}

class RegularUser implements User {
  permissions = ["read"];
}

const userFactory: Record<UserRole, () => User> = {
  admin: () => new Admin(),
  user: () => new RegularUser(),
};

function createUser(role: UserRole): User {
  const creator = userFactory[role];
  if (!creator) throw new Error("Unknown role");
  return creator();
}
```
