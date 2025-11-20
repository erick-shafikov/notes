# Функции

Типизация функций

```ts
interface TwoNumberCalculation {
  (x: number, y: number): number; //с помощью интерфейса, так же можно type TwoNumberCalculation = {}
}

type TwoNumberCalc = (x: number, y: number) => number; //с помощью типа
const add: TwoNumberCalculation = (a, b) => a + b;
const subtract: TwoNumberCalc = (a, b) => a - b;

// Типизация параметров
type DescribableFunction = {
  //функцию можно определить как объект, так как у функции могут быть свойства
  description: string;
  (someArg: number): boolean;
};
function doSomething(fn: DescribableFunction) {
  //определяем тип параметра как объект-функции
  console.log(fn.description + " returned " + fn(6)); //вызываем и свойство функции и взываем саму функцию
}

// определяем
function myFunc(someArg: number) {
  return someArg > 3;
}
// добавляем свойство
myFunc.description = "default description";
doSomething(myFunc);
```

## Опциональные параметры

```ts
function f(x?: number) {
  // ...
}
f(); // OK
f(10); // OK
```

# перегрузка методов

Последняя должна быть обобщающая

```ts
// проблема вернет string | number
function getLength(val: string: any[]){
  if(typeof val === 'string'){
    return `${val.split(' ').length} words`
  }

  return val.length
}

const numberOfWords = getLength('aaa bbb ccc')
numberOfWords.length //ошибка так как length нет на number
```

```ts
function getLength(val: any[]):number;
function getLength(val: string):string;
function getLength(val: string: any[]){
  if(typeof val === 'string'){
    return `${val.split(' ').length} words`
  }

  return val.length
}

const numberOfWords = getLength('aaa bbb ccc')
numberOfWords.length //ок, знает что string
```
