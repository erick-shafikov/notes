# Декоратор параметра

```ts
function Param(
  target: Object,
  propertyKey: string,
  index: number //индекс конкретного параметра
) {
  console.log(propertyKey, index);
}

// @logger
// @Component(1)
// export class User {
//  @Prop id: number;
// @Method

  updatedId(@Param newId: number) {
    this.id = newId;
    return this.id;
  }

// }
// console.log(new User().id);
// console.log(new User().updatedId(2));
```
