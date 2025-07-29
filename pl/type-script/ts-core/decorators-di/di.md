# DI1

- DI через перебрасывание в конструктор
- DI через использование в конкурентном методе класса
- через интерфейсы

# MetadataReflection

npm i reflect-metadata – установка

в tscconfig.json должны быть включены

```json
"experimentalDecorators": true, /* Enable experimental support for TC39 stage 2 draft decorators. */
"emitDecoratorMetadata": true, /* Emit design-type metadata for decorated declarations in source files. */
```

```ts
Reflect.defineMetadata(
  name: string,
  value,
  SomeClass.prototype,
  propName: string
)

Reflect.getMetadata(name, SomeClass.prototype, propName)
```

```ts
import "reflect-metadata"; //импорт без какого-либо деструктурировано

function Injectable(key: string) {
  return (target: Function) => {
    Reflect.defineMetadata(key, 1, target); //определяем метаданные первый параметр - переменная в мета, второй - значение переменной, третий - цель, где будет определенна мета-переменная (для класса, свойства)
    const meta = Reflect.getMetadata(key, target);
    console.log(meta);
  };
}
function Prop(target: Object, name: string) {}

@Injectable("C")
export class C {
  @Prop prop: number;
}
@Injectable("D")
export class D {
  //constructor(@Inject('C') c: C){}
}
```

Пример 2

```ts
@Injectable("name-of-injection")
class SomeInjectableClass {}

@Inject("name-of-injection")

function Injectable(label:string){
  return function(target: any, context: any){
    // target - класс
  }
}

function Inject(bane: string){
  return function (target: any, context: any){

  }
}
```

# DTO. data transfer object

это файлы, которые содержат в себе классы

```ts
login(req: Request<{}, {}, UserLoginDto>, res: Response, next: NextFunction): void

// где
export class UserLoginDto {
    email: string;
    password: string;
}
```
