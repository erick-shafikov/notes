# is

позволяет явно приравнять к типу тот или иной объект/переменную

```ts
function isFish(pet: Fish | Bird): pet is Fish {
  //
  //сообщаем функции, что pet - Это Fish
  return (pet as Fish).swim !== undefined; //вызываем метод swim у pet, если он есть, то вернутся true
} // применение
let pet = getSmallPet();
if (isFish(pet)) {
  pet.swim();
} else {
  pet.fly();
}
```
