# Расширяемый generic

```ts
//c помощью ключевого слова extends мы указываем TS, что generic Type содержит поле length
function longest<Type extends { length: number }>(a: Type, b: Type) {
  if (a.length >= b.length) {
    return a;
  } else {
    return b;
  }
} // longerArray is of type 'number[]'
const longerArray = longest([1, 2], [1, 2, 3]);
// longerString is of type 'alice' | 'bob'
const longerString = longest("alice", "bob");
// Error! Numbers don't have a 'length' property
const notOK = longest(10, 100);
// Argument of type 'number' is not assignable to parameter of type '{ length: number; }'.
```

```ts
class Vehicle {
  //объект
  run!: number;
}

function kmToMiles<T extends Vehicle>(vehicle: T): T {
  //без расширения не определит тип
  vehicle.run = vehical.run / 0.62;
  return vehicle;
}

class LCV extends Vehicle {
  capacity!: number;
}

const vehicle = kmToMiles(new Vehicle());
const lvc = kmToMiles(new LCV());

kmToMiles({ run: 1 }); // тоже сработает так как интерфейс схож

function logId<T extends string | number>(id: T): T {
  console.log(id);
  return id;
}
```
