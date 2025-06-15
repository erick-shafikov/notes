# props

```js
function Welcome(props) {
  return <h1>Привет, {props.name}</h1>;
}
const element = <Welcome name="Алиса" />;
ReactDOM.render(element, document.getElementById("root"));
```

!!!Примечание: Всегда называйте компоненты с заглавной буквы. Если компонент начинается с маленькой буквы, React принимает его за DOM-тег. Например, div это div-тег из HTML, а Welcome это уже наш компонент Welcome, который должен быть в области видимости.

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

# children prop

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

# keys prop

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
