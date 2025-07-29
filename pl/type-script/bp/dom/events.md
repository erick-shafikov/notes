# для событий

```ts
const form = document.getElementById("signup-form") as HTMLFormElement;
form.addEventListener("submit", (e: Event) => {
  console.log(e.target); // ОШИБКА: Свойство 'target' не существует у типа 'Event'. Может, вы имели ввиду 'target'?
});
```
