# typeof

typeof из js возвращает строковое представление типа объекта. В ts typeof позволяет достать тип ts

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
```

# typeof и keyof

```ts
//совмещение typeof и keyof
const user = {
  name: "Vasya",
};

// type keyofUser = keyof user; так нельзя так как user это не тип
type keyofUser = keyof typeof user; //type keyofUser = "name"
```

# keyof typeof и enum

```ts
enum Direction {
  Up,
  Down,
}
type d = keyof typeof Direction; //type d = "Up" | "Down" названия enum'ов
```
