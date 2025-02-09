# REACT

- packages - основная директория с исходниками react, содержит более 35 пакетов, в которых хранится весь исходный код, основные из них:

- react - содержит весь код для создания компонентов, react-render позволяет отрисовывать дерево в react-dom или в любую другую среду
- react-reconciler - пакет для сравнения виртуальных деревьев react
- react-dom и react-native-renderer библиотеки для рендера в различных средах, где может работать react
- react-devtools - несколько пакетов для отладки react приложений на более высоком уровне, чем просто javascript код

- React Element – это объектное представление каждого компонента react
  {type: «html-tag», prop: any, children: ReactNode[]}, они иммутабельные.
- Есть два типа элемента React DOM Element – то, что можно представить в виде html-тега и React Element Component – обертка для совокупности React DOM Element-ов, которые в свою очередь превращаются в DOM дерево в последствии рендера
- Входная точка – ReactDOM.render(), обеспечивает функционирование механизма сопоставления (reconciliation)
  При повторном вызове ReactDOM.render(), вызывается способ Reconciliation, сравниваются типы элементов и их значения. Списки определяются с помощью key. Сопоставление происходит в две фазы чтобы мы не видели подёргивания интерфейса, делает изменения в 2 фазы
- React не использует requestIdleCallback (для вызова функции в зависимости от свободных ресурсов системы), а использует библиотеку scheduler

# Дерево элементов, fiber

- В React два дерева – дерево элементов и fiber.
- В дереве элементов находится UI. В fiber – находится состояние, пропсы, предыдущие состояния и пропсы, которые позволяют хранит дополнительную информацию о компонентах.

Fiber – связанный лист. JS – объект.

```js
const fiberNode = {
  stateNode,
  memoizedProps,
  memoizedState,
  child,
  parent,
  sibling;
}
```

В fiber два типа связи – дочерние и соседские, то есть все элементы знаю о своих родителях и соседях. Самая верхняя node – Host root.

- Первая фаза: «Эффекты»

- Каждый элемент производит “работы” - эффекты (запросы данных, подписки, изменение dom), эффекты тоже лист. Все эффекты они приоритизированы. Они делят на два типа деревьев – current tree, а в том, котором происходят изменения VIP-tree. Таким образом есть 2 процесса rendering и reconciliation.

- Reconciliation можно отменить, так как могли произойти изменения во время более ранних изменений.
  React сравнивает типы элементов и если меняется тип, то элемент и его дочерние элементы демонтируются.
  fiber node мутабельные, dom node – иммутабельные

Вторая фаза «Commit» Здесь выполняются эффекты и меняется дерево элементов, фаза непрерываемая

# concurrent mode

CCMode – это режим работы react-приложения, когда приложение остается отзывчивым и не блокируется, даже если на фона происходят большие вычисления. Аналогично git ведутся работы над разными изменениями в dom. Пример с вводом текста в поле input, при котором блокируется UI. debounce и throttling могут так же вызывать. Смысл в том, что react может работать параллельно над несколькими обновлениями состояния. Данный режим изначально назывался асинхронные компоненты, конкурентный режим, теперь concurrent features

Хуки которые использует CC-mode useTransition, startTransition, useDeferredValue

Приоритетные задачи на обновления:

- HIGH: useState, useReducer, useSyncExternalStore
- LOW: все остальное

Принципы выполнения задач:

- высокоприоритетные такси прерывают низкоприоритетные
- выполнение функции компонента всегда доходит до return, компонент нельзя прервать
- низкоприоритетные запоминают свое состояние

## Виртуальный DOM

Виртуальный DOM –это библиотека JS, которая позволяет работать с настоящим DOM без поиска элементов, работы с атрибутами и прочих рутинных операций. Представляет собой пользовательский интерфейс в виде объекта. Основной элемент – это createRoot(), который активирует виртуальный DOM, далее каждый элемент создается с помощью createElement(тип элемента, атрибуты, дети)

## Методы жизненного цикла

Общий принцип

1. конструктор – вызывается один раз при построении компонента
2. если присваивать пропсы в конструкторе, то они будут доступны только при инициализации компонента, так как конструктор вызывается один раз
3. Каждый раз, при перерисовке - getDerivedStateFromProps(props, state){} – статичный метод, работает при каждом обновлении нужен при работе с пропсами и состояниями, возвращает либо новое состояние либо null, срабатывает каждый раз, отслеживает изменение пропсов
4. render’ы
5. ComponentDidMount – вызывается после отрисовки, не работает при обновлении – компонент выведен и закончил отрисовку, не работает при обновлении
6. componentDidUpdate – вызывается каждый раз при обновлении компонента, не вызывается при первом рендере

7. Монтирование
   При создании экземпляра компонента и его вставке в DOM, следующие методы вызываются в установленном порядке:
   - constructor()
   - static getDerivedStateFromProps(props, state)
   - UNSAFE_componentWillMount() - устарел
   - render()
   - componentDidMount() – подписки на события и сетевые запросы === useEffect(() => {}, [])
8. Обновление
   Обновление происходит при изменении пропсов или состояния. Следующие методы вызываются в установленном порядке при повторном рендере компонента:
   - UNSAFE_componentWillReceiveProps() – устарели
   - static getDerivedStateFromProps(props, state)
   - shouldComponentUpdate() – если вернет false, render() не вызывается, но это не предотвратит рендер дочерних компонентов и их состояния
   - UNSAFE_componentWillUpdate
   - render()
   - getSnapshotBeforeUpdate(prevStet, prevProps) – вызывается перед фиксированием, полезно для простановки прокрутки
   - componentDidUpdate(prevState, newState, this.props, this.state) – сразу после обновления === useEffect(() => {})

Каждое обновление происходит в две фазы – рендер (React указывает что должно быть на экране), коммит – React применяет эти изменения в DOM

3. Размонтирование
   componentWillUnmount() – отписка от событий, этот метод вызывается до удаления компонента из DOM === useEffect(() => { return () => {} }, [])

4. Обработка ошибок
   Следующие методы вызываются, если произошла ошибка в процессе рендеринга, методе жизненного цикла или конструкторе любого дочернего компонента.

```js
static getDerivedStateFromError(error) {
// Обновите состояние так, чтобы следующий рендер показал запасной интерфейс.
  return { hasError: true };
}

componentDidCatch()
```

component.forceUpdate(callback) – для повторного рендеринга, если он зависит от каких либо других параметров пропустит shouldComponentUpdate()

![react-lifecycle](/assets/react/react-lifecycle.png)

## повторный рендер элементов

- Изменение props
- Изменение state
- Изменился родитель (предотвращается memo)
- Контекст

# children. composition

проп children используется для компонентов, в которых содержимое неопределенно до рендеринга

```js
function FancyBorder(props) {
  return (
    <div className={"FancyBorder FancyBorder-" + props.color}>
      {/*задано в css*/}
      {props.children} 
    </div>
  );
}
function WelcomeDialog() {
  return (
    <FancyBorder color="blue">
      {/*все, что ниже будет являться props.children */}
      <h1 className="Dialog-title">
        {/*задано в css */}
        Добро пожаловать
      </h1>
      <p className="Dialog-message">
        {/*задано в css*/}
        Спасибо, что посетили наш космический корабль!  
      </p>
       
    </FancyBorder>
  );
}
```

# conditional rendering

Условный рендеринг

```jsx
function UserGreeting(props) {
  return <h1>Welcome back!</h1>;
}
function GuestGreeting(props) {
  return <h1>Please sign up.</h1>;
}
function Greeting(props) {
  const isLoggedIn = props.isLoggedIn;
  if (isLoggedIn) {
    return <UserGreeting />;
  }
  return <GuestGreeting />;
}
ReactDOM.render(
  // Try changing to isLoggedIn={true}:
  <Greeting isLoggedIn={true} />,
  document.getElementById("root")
);
```

<!-- JSX----------------------------------------------------------------------------------------------------------------------------------->

# JSX

```jsx
import React, { Component } from "react";
import "./App.css";
class App extends Component {
  render() {
    //обычный метод рендеринга
    /* return (
<div className="App">
Hello
</div>
); */
    return React.createElement(
      //альтернативный вариант
      "div", //
      { className: "App" },
      React.createElement("h1", null, "Hello World")
    );
  }
}
export default App;
```

```jsx
//
const element = <h1>Привет, мир!</h1>; //- элемент React

const element = <h1 className="greeting"> Привет, мир! </h1>;

// аналогично
const element = React.createElement(
  "h1",
  { className: "greeting" },
  "Привет, мир!"
);
```

# keys

Рендер списка от 1 до 5

```js
const numbers = [1, 2, 3, 4, 5];
const listItems = numbers.map(
  (number) => <li>{number}</li> //возвращаем компонент li для каждого элемента
);
ReactDOM.render(
  <ul>{listItems}</ul>, //упаковываем в список ul
  document.getElementById("root")
);
```

```js
function NumberList(props) {
  const numbers = props.numbers;
  const listItems = numbers.map((number) => (
    <li key={number.toString()}>
      //определение ключа всегда происходит в map-функции
      {number}
    </li>
  ));
  return <ul>{listItems}</ul>;
}
export default NumberList;
//index.js
const numbers = [1, 2, 3, 4, 5]; //массив отправится в пропсы
ReactDOM.render(
  <NumberList numbers={numbers} />,
  document.getElementById("root")
);
```

```js
// включение результата map внутри компонента

function NumberList(props) {
  const numbers = props.numbers;
  return (
    <ul>
      {numbers.map((number) => (
        <ListItem key={number.toString()} value={number} />
      ))}
    </ul>
  );
}
```

```js
function App() {
  return (
    <div>
      <Header /> {/*статичный компонент */}
      {notes.map((item) => (
        <Note title={item.title} content={item.content} key={item.key} /> //notes – экспортированный массив объектов
      ))}
      <Footer /> {/*статичный компонент */}
    </div>
  );
}

// export default App;
//Note.jsx
function Note(props) {
  return (
    <div className="note">
      <h1>{props.title}</h1>
      <p>{props.content}</p>
    </div>
  );
}
// export default Note;
```

## keys. BP

Что бы вызывать рендер компонента каждый раз в key нужно передавать произвольное число

<!-- props-------------------------------------------------------------------------------------------------------------------------------->

## props

```js
function Welcome(props) {
  return <h1>Привет, {props.name}</h1>;
}
const element = <Welcome name="Алиса" />;
ReactDOM.render(element, document.getElementById("root"));
```

!!!Примечание: Всегда называйте компоненты с заглавной буквы. Если компонент начинается с маленькой буквы, React принимает его за DOM-тег. Например, div это div-тег из HTML, а Welcome это уже наш компонент Welcome, который должен быть в области видимости.

# Styling

## modules

```scss
.container_span {
  text-shadow: 4px 4px 3px rgb(154, 158, 161);
  text-decoration: underline;
  font-family: 'Popins';
}

h1{
  font-family: 'Popins';  
}
```

```jsx
import "./App.css";

function App() {
  return (
    <div>
      <h1>Hello</h1>
      <span className="container_span">This is a book app</span>
    </div>
  );
}
```

Модули с уникальным префиксом

```jsx
import styles from "./App.module.css";

function App() {
  return (
    <div>
      <h1>Hello</h1>
      <span className={styles.container_span}>This is a book app</span>
    </div>
  );
}
```

```scss
.green {
  border: 1px solid green;
}
.input.red {
  border: 1px solid red;
}
```

```js
//Car.js
import classes from "./Car.module.css";
const inputClasses = [classes.input]; //inputClasses - это массив из одного объекта стиля [Car_префикс1, Car_префикс2], с использованием
console.log(classes.green);
if (this.props.name !== "") {
  //если не пустая строка
  inputClasses.push(classes.green); //classes.green : Car_green__1BE2p
} else {
  inputClasses.push(classes.red);
}
if (this.props.name.length > 4) {
  inputClasses.push(classes.bold);
}

<input className={inputClasses.join(" ")} />; // теперь соединяем в строчку
```

# SVG

```ts
//для инициализации в TS
declare module '*.svg' {
  import React from 'react';
  const SVG: React.VFC<React.SVGProps<SVGElement>>;
  export default SVG;
}

//в WP конфиге:
module: {
  rules: [
    {
      test: /\.svg$/,
      use: ['@svgr/webpack']
    }
  ]
},

```

# BP

# BP. forms

```js
// Пример формы в виде функционального компонента

function App() {
  const [name, setName] = useState("");
  const [headingText, setHeading] = useState("");
  function handleChange(e) {
    setName(e.target.value);
  }
  const onClickHandler = (e) => {
    setHeading(name);
    e.preventDefault();
  };
  return (
    <div className="container">
      <h1>Hello {headingText}</h1> 
      <form onSubmit={onClickHandler}>
         
        <input
          onChange={handleChange}
          type="text"
          placeholder="What's your name?"
          value={name}
        />
        <button type="submit">Submit</button> 
      </form>
    </div>
  );
}
```
