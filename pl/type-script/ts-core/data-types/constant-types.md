c as const | без as const

```ts
let roles = ["admin", "guest", "editor"] as const;

roles.push("max"); //ошибка | ок
roles[0]; // 'admin' | string
```
