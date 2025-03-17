# объекты со свойством next

Создание итерируемого объекта

```js
function makeIterator(array) {
  var nextIndex = 0;

  return {
    next: function () {
      return nextIndex < array.length
        ? { value: array[nextIndex++], done: false }
        : { done: true };
    },
  };
}

var it = makeIterator(["yo", "ya"]);
console.log(it.next().value); // 'yo'
console.log(it.next().value); // 'ya'
console.log(it.next().done); // true
```

<!-- генераторы ------------------------------------------------------------------------------------------------------------------------------>

# генераторы

синтаксис

```js
function* generatorSequence() {
  //нет разницы function* f() или function *f()
  yield 1;
  yield 2;
  return 3;
}
```

Когда вызывается функция генератор, она не выполняет свой код, а возвращает специальный объект «генератор»

```js
let generator = generatorSequence();
alert(generator); //[object Generator]
```

## next

- Основной метод – next().
- При вызове он запускает выполнение кода до ближайшей инструкции yield <значение>, если оно отсутствует, то оно предполагается равным undefined.
- По достижении yield выполнение функции приостанавливается, а соответствующее значение возвращается во внешний код
  Результатом next() является объект с двумя свойствами { value: значение из yield, done : true/false}

```js
let generator = generatorSequence();
let one = generator.next(); //{value: 1, done: false}
let two = generator.next(); //{value: 2, done: false}
let three = generator.next(); //{value: 3, done true}, каждый следующий будет возвращать {done: true}
```

## return

возвращает полученное значение и останавливает генератор

```js
function* gen() {
  yield 1;
  yield 2;
  yield 3;
}

var g = gen();
g.next(); // { value: 1, done: false }
g.next(); // { value: 2, done: false }
g.next(); // { value: 3, done: false }
g.next(); // { value: undefined, done: true }
g.return(); // { value: undefined, done: true }
g.return(1); // { value: 1, done: true }
```

## Перебор генераторов

Возвращаемые значения можно перебирать через for...of

```js
function* generateSequence() {
  yield 1;
  yield 2;
  return 3; //если поменяем на yield 3, то цикл for of выведет 1,2,3
}
let generator = generateSequence();
for (let value of generator) {
  alert(value); //1,2
}

// Так как генераторы перебираемые объекты, то можно использовать связанную с ними функциональность
let sequence = [0, ...generateSequence()]; //0,1,2,3,
```

## Использование генераторов для перебираемых объектов

```js
let range = {
  from: 1,
  to: 5,
  [Symbol.iterator]() {
    //for of вызывает этот метод один раз в самом начале
    return {
      current: this.from,
      last: this.to,
      next() {
        if (this.current <= this.last) {
          return { done: false, value: this.current++ };
        } else {
          return { done: true };
        }
      },
    };
  },
};

for (let value of range) {
  alert(value);
}

let range = {
  from: 1,
  to: 5,
  *[Symbol.iterator]() {
    //краткая запись [Symbol.iterator]:function*()
    for (let value = this.from; value <= this.to; value++) {
      yield value;
    }
  },
};
```

## Композиции генераторов

Функция для генерации последовательности чисел

```js
function* generateSequence(start, end) {
  for (let i = start; i <= end; i++) {
    yield i;
  }
}
// Для генерации последовательности сначала цифры 0, ...9  (c кодами символов 48...57)

function* generateSequence(start, end){
  for (let i = start, i <= end, i++) {
  yield i
  }
}

// Функция для генерации строки формата 0…9A…Za…z

function* generatePasswordsCodes() {
  yield* generateSequence(48, 57);
  yield* generateSequence(65, 90);
  yield* generateSequence(97, 122);
}

let str = "";

for (let code of generatePasswordCodes()) {
  str += String.fromCharCodes(code);
}

alert(str); //0…9A…Za…z

// тоже самое, но через вложенные циклы
// function* generateAlphaNum() {
// for (let i = 48; i <= 57, i++) yield i
// for (let i = 65; i <= 90, i++) yield i
// for (let i = 97; i <= 122, i++) yield i
// }

let str = "";

for (let code of generateAlphaNum()) {
  str += String.fromCharCodes(code);
}

alert(str);
```

## yield

yield не только возвращает результат наружу, но и может передавать значение извне в генератор,
синтаксис: generator.next(arg)

```js
function* gen() {
  let result = yield "2+2 = ?";
  alert(result);
}

let generator = gen();

let question = generator.next().value; //yield возвращает значение
generator.next(4); //передача в генератор, как результат текущего yield, с последующим выводом  результата.

setTimeout(() => generator.next(4), 1000); //код не обязан немедленно вызывать next(4), генератор подождет

// в отличие от обычных функций, генератор может обмениваться результатами с вызывающим кодом
function* gen() {
  let ask1 = yield "2 + 2 = ?";
  alert(ask1); // (2) 4
  let ask2 = yield "3 * 3 = ?";
  alert(ask2); // 9
}

let generator = gen();
alert(generator.next().value); //(1) 2+2=? вызывает yield в первой строчке, останавливается на этом
alert(generator.next(4).value); //3*3 = ? присваивает к yield в первой строчке значение 4, выводя  значение первым alert" ом и переходит к вызову yield на следующей строчке
alert(generator.next(9).done); //true присваивает предыдущему yield 9, выполняя второй ask, в поисках
```

```js
function* gen() {
  while (true) {
    var value = yield null;
    console.log(value);
  }
}

var g = gen();
g.next(1);
// "{ value: null, done: false }"
g.next(2);
// 2
// "{ value: null, done: false }"
```

## generator.throw

для того чтобы передать ошибку в yield нужно вызвать generator.throw(err). В этом случае исключение err возникнет на строке с yield. Перехват ошибки:

```js
function* gen() {
  try {
    let result = yield "2+2=?"; //
    alert("Выполнение программы не дойдет до этого alert");
  } catch (e) {
    alert(e);
  }
}

let generator = gen();
let question = generator.next().value;
generator.throw(new Error("Нет ответа в базе данных")); // ошибка проброшенная здесь даст возможность  только выполнить, то что рядом с yield

// перехват ошибки во внешнем коде
function* generate() {
  let result = yield "2+2=?";
}
let generator = generate();
let question = generator.next().value;
try {
  generator.throw(new Error("Ответ не найден"));
} catch (e) {
  alert(e);
}
```

```js
async function* range(start, end) {
  //функция генератора range возвращает асинх объект
  for (let i = start; i <= end; i++) {
    yield Promise.resolve(i); //возвращает Promise {i}
  }
}
(async () => {
  //так как итерируемые объекты - асинхронные и итерируемы
  const gen = range(1, 3); //gen - асинхронный объект
  for await (const item of gen) {
    //если убрать await - ошибка
    console.log(item);
  }
})();
```

<!-- Асинхронные итераторы ---------------------------------------------->

# Асинхронные итераторы

- Symbol.asyncIterator вместо Symbol.Iterator
- next() должен возвращать промис
- Перебор осуществляется через for await (let item of iterable)

```js
let range = {
  from: 1,
  to: 2,
  [Symbol.asyncIterator]() {
    //Чтобы сделать объект асинхронно итерируемым, он должен иметь метод
    // [Symbol.asyncIterator]
    return {
      current: this.from,
      last: this.to,
      async next() {
        //метод должен возвращать объект с методом next(), который должен возвращать промис, метод next() не  обязательно должен быть async, он может быть и обычным методом, возвращающим промис, но async позволяет  использовать await
        await new Promise((resolve) => setTimeout(resolve, 2000)); //так как делаем паузу в 2 секунды
        if (this.current <= this.last) {
          return {
            done: false,
            value: this.current++,
          };
        } else {
          return { done: true };
        }
      },
    };
  },
};

(async () => {
  for await (let value of range) {
    //для итерации мы используем for await (let value of range). Он вызовет
    // range[Symbol.asyncIterator]() один раз, а затем его метод next() для получении значений.
    alert(value);
  }
})();
```

```js
async function* generateSequence(start, end) {
  for (let i = start; i <= end; i++) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    yield i;
  }
}

(async () => {
  let generator = generateSequence(1, 5);

  for (let value of generator) {
    alert(value);
  }
})();

// метод generator.next() теперь тоже асинхронный и возвращает промисы
result = await generator.next();
```

## Асинхронно перебираемые объекты с помощью асинхронного генератора

```js
let range = {
  from: 1,
  to: 5,

  async *[Symbol.asyncIterator]() {
    for (let value = this.from; value <= this.to; value++) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      yield value;
    }
  },
};

(async () => {
  for await (let value of range) {
    alert(value);
  }
})();
```

```js
async function* foo() {
  yield await Promise.resolve("a");
  yield await Promise.resolve("b");
  yield await Promise.resolve("c");
}

let str = "";

async function generate() {
  for await (const val of foo()) {
    str = str + val;
  }
  console.log(str);
}

generate();
```

<!-- BPs ---------------------------------------------------------------->

## BP

- запрос на url в виде https://api.github.com/repos/repo/commits
- в ответ придет JSON с 30 коммитами, а также ссылка на следующую страницу в заголовке Link
- Затем нужно использовать эту ссылку для следующего запроса, чтобы получить дополнительную порцию
  коммитов

```js
async function* fetchCommits(repo) {
  let url = "https://api.github.com/repos/${repo}/commts";
  while (url) {
    const response = await fetch(url, {
      headers: { "User-Agent": "Our script" },
    });

    const body = await response.json();
    let nextPage = response.headers.get("Link").match(/<(.*?)>; rel="next"/);
    nextPage = nextPage && nextPage[1];
    url = nextPage;
    for (let commit of body) {
      yield commit;
    }
  }
}
// используем

(async () => {
  let commit = 0;
  for await (const commits of fetchCommits("comments.com/comments")) {
    console.log(commit.author.login);
    if (++count == 100) {
      break;
    }
  }
})();
```

## Bps: числа фибоначчи на генераторах

```js
function* fibonacci() {
  var fn1 = 1;
  var fn2 = 1;
  while (true) {
    var current = fn2;
    fn2 = fn1;
    fn1 = fn1 + current;
    var reset = yield current;
    if (reset) {
      fn1 = 1;
      fn2 = 1;
    }
  }
}

var sequence = fibonacci();
console.log(sequence.next().value); // 1
console.log(sequence.next().value); // 1
console.log(sequence.next().value); // 2
console.log(sequence.next().value); // 3
console.log(sequence.next().value); // 5
console.log(sequence.next().value); // 8
console.log(sequence.next().value); // 13
console.log(sequence.next(true).value); // 1
console.log(sequence.next().value); // 1
console.log(sequence.next().value); // 2
console.log(sequence.next().value); // 3
```
