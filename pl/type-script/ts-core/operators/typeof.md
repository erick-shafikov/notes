# typeof

```ts
let strOnNum: string | number;

if (Math.random() > 0.5) {
  strOnNum = 5;
} else {
  strOnNum = "str";
}

if (typeof strOnNum === "string") {
  console.log(strOnNum);
} else {
  console.log(strOnNum);
}

let strOnNum2: typeof strOnNum; //let strOnNum2: string | number
//совмещение typeof и keyof
const user = {
  name: "Vasya",
};

// type keyofUser = keyof user; так нельзя так как user это не тип
type keyofUser = keyof typeof user; //type keyofUser = "name"
enum Direction {
  Up,
  Down,
}
type d = keyof typeof Direction; //type d = "Up" | "Down" названия enum'ов
```
