# Перегрузка методов

```ts
class User {
  skills: string[];
  addSkill(skill: string): void; //в зависимости от типа аргумента позволяет реализовать разный функционал
  addSkill(skill: string[]): void;
  addSkill(skill: string | string[]): void {
    if (typeof skill === "string") {
      this.skills.push(skill);
    } else {
      this.skills.concat(skill);
    }
  }
} //перегрузка функций
function run(distance: string): string;
function run(distance: number): number;
function run(distance: number | string): number | string {
  if (typeof distance === "number") {
    return 1;
  } else {
    return "";
  }
}
```
