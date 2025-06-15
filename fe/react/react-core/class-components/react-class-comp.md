!!!жизненный цикл перенести в эту папку

# Композиция

проп children используется для компонентов, в которых содержимое неопределенно до рендеринга

```jsx
function Dialog(props) {
  return (
    <>
      <h1 className="Dialog-title">{props.title}</h1>
      <p className="Dialog-message">{props.message}</p>
      {props.children}
    </>
  );
}

class SignUpDialog extends React.Component {
  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.handleSignUp = this.handleSignUp.bind(this);
    this.state = { login: "" };
  }

  handleChange(e) {
    this.setState({ login: e.target.value });
  }
  handleSignUp() {
    alert(`Добро пожаловать на борт, ${this.state.login}!`);
  }
  render() {
    return (
      <Dialog
        title="Программа исследования Марса"
        message="Как к вам обращаться?"
      >
        <input value={this.state.login} onChange={this.handleChange} />
        <button onClick={this.handleSignUp}>Кнопка</button>
      </Dialog>
    );
  }
}
```

# conditional rendering

Он будет рендерить либо <LoginButton />, либо <LogoutButton /> в зависимости от текущего состояния. А ещё он будет всегда рендерить <Greeting /> из предыдущего примера.

```jsx
class LoginControl extends React.Component {
  constructor(props) {
    super(props);
    this.handleLoginClick = this.handleLoginClick.bind(this);
    this.handleLogoutClick = this.handleLogoutClick.bind(this);
    this.state = { isLoggedIn: false };
  }
  handleLoginClick() {
    this.setState({ isLoggedIn: true });
  }
  handleLogoutClick() {
    this.setState({ isLoggedIn: false });
  }
  render() {
    const isLoggedIn = this.state.isLoggedIn;
    let button;
    if (isLoggedIn) {
      button = <LogoutButton onClick={this.handleLogoutClick} />;
    } else {
      button = <LoginButton onClick={this.handleLoginClick} />;
    }
    return (
      <div>
        <Greeting isLoggedIn={isLoggedIn} />
        {button}
      </div>
    );
  }
}
```

# context

значение пропса theme должно быть передано вниз из T

```jsx
class Example extends React.Component {
  render() {
    return <Toolbar theme="dark" />;
  }
}
//переедает проп theme в toolbar

function Toolbar(props) {
  return (
    <div>
      <ThemedButton theme={props.theme} />
    </div>
  );
}
//переедает проп theme в ThemeButton
class ThemedButton extends React.Component {
  render() {
    return <Button theme={this.props.theme} />;
  }
} //переедает проп theme в Button
class Button extends React.Component {
  render() {
    return <button>{this.props.theme}</button>;
  }
}
```

с использованием контекста

```js
const ThemeContext = React.createContext("light");
//создаем контекст с помощью createContext
class Example extends React.Component {
  render() {
    return (
      //передаем пробрасываемое значение
      <ThemeContext.Provider value="blue">
        <Toolbar />
      </ThemeContext.Provider>
    );
  }
}
function Toolbar() {
  return (
    <div>
      <ThemedButton />
    </div>
  );
}
class ThemedButton extends React.Component {
  render() {
    console.log(this.context);
    return <Button />;
  }
}
class Button extends React.Component {
  static contextType = ThemeContext;
  //фиксируем местонахождение
  render() {
    return <button>{this.context}</button>; //используем
  }
}
```

```jsx
const ThemeContext1 = React.createContext({ color: "green" }); //(1.1)создаем объект ThemeContext
const ThemeContext2 = React.createContext(); //создаем второй объект контекста (1.2)
class Example extends React.Component {
  render() {
    return (
      //отправляем первый (2.1)
      <ThemeContext1.Provider value={{ color: "brown" }}>
        <Toolbar />
      </ThemeContext1.Provider>
    );
  }
}
function Toolbar() {
  return (
    <div>
      <ThemedButton />
    </div>
  );
}

class ThemedButton extends React.Component {
  static contextType = ThemeContext1;
  render() {
    //отправляем второй (2.2)
    //       context: {this.context.color}
    return (
      <div
        _style={{
          backgroundColor: `${this.context.color}`,
          height: 100,
          color: "yellow",
        }}
      >
        <ThemeContext2.ProviderValue value="blue">
          <Button />
        </ThemeContext2.ProviderValue>
      </div>
    );
  }
}
class Button extends React.Component {
  static contextType = ThemeContext2; //(3) идет в паре с (4)
  render() {
    return (
      <button _style={{ backgroundColor: `${this.context}` }}>
        {this.context}
      </button>
    ); //(5) достаем значение контекста
  }
}
```

# Events handler

```jsx
// HTML: <button onclick="activateLasers()"> Активировать лазеры </button>
// JSX: <button onClick={activateLasers}> Активировать лазеры </button>

// Пример привязки события через конструктор в классовом компоненте

class Toggle extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };
    this.handleClick = this.handleClick.bind(this); //необходимо для работы в колбэке
  }
  handleClick() {
    this.setState((prevState) => ({
      isToggleOn: !prevState.isToggle, //меняем на противоположное состояние
    }));
  }
  render() {
    return (
      <button onClick={this.handleClick}>
        {/*приравнивание без аргументов*/}
        {this.state.isToggleOn ? "ON" : "OFF"}
        {/*в зависимости от состояния отображать ON или OFF*/}
      </button>
    );
  }
}
```

```jsx
class LoggingButton extends React.Component {
  // Такой синтаксис гарантирует, что `this` привязан к handleClick.
  // Предупреждение: это экспериментальный синтаксис
  handleClick = () => {
    //меняем обычное поле на стрелочную функцию
    console.log("значение this:", this);
  };
  render() {
    return <button onClick={this.handleClick}>Нажми на меня</button>;
  }
}

// с помощью коллбека в обработчике события
class LoggingButton extends React.Component {
  handleClick() {
    console.log("значение this:", this);
  }
  render() {
    // Такой синтаксис гарантирует, что `this` привязан к handleClick.
    return <button onClick={() => this.handleClick()}>Нажми на меня</button>;
  }
}
```

## Передача аргументов в обработчики событий

```js

<button onClick={(e) => this.deleteRow(id, e)}>Удалить строку</button>
<button onClick={this.deleteRow.bind(this, id)}>Удалить строку</button>
```

В обоих случаях аргумент e, представляющий событие React, будет передан как второй аргумент после идентификатора. Используя стрелочную функцию, необходимо передавать аргумент явно, но с bind любые последующие аргументы передаются автоматически.

!!!в React нельзя предотвратить обработчик события по умолчанию, вернув false. Нужно явно вызвать preventDefault

# forms

!!!Обновление состояния форм только через setstate
!!!сброс действия только через event.preventDefault()

```js
class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = { value: "" }; //состояние
    this.handleChange = this.handleChange.bind(this); //изменение состояния при вводе
    this.handleSubmit = this.handleSubmit.bind(this); //изменение состояния при submit
  }
  handleChange(event) {
    this.setState({ value: event.target.value }); //перезапись состояния при каждом вводе
  }
  handleSubmit(event) {
    alert("Отправленное имя: " + this.state.value); //при submit ввывести значение поля
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          Имя:
          <input
            type="text"
            value={this.state.value}
            onChange={this.handleChange}
          />
        </label>
        <input type="submit" value="Отправить" />
      </form>
    );
  }
}
```

Тег `<textarea>`

```js
class EssayForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      value: "Будьте любезны, напишите сочинение о вашем любимом DOM-элементе.",
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }
  handleChange(event) {
    this.setState({ value: event.target.value });
  }
  handleSubmit(event) {
    alert("Сочинение отправлено: " + this.state.value);
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          Сочинение:
          <textarea value={this.state.value} onChange={this.handleChange} />
        </label>
        <input type="submit" value="Отправить" />
      </form>
    );
  }
}
```

## Формы. `<select>`

```js
class FlavorForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = { value: "coconut" }; //выбран Кокос
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }
  handleChange(event) {
    this.setState({ value: event.target.value });
  }
  handleSubmit(event) {
    //при submit – alert выбранного пункта сохраненного в state
    alert("Ваш любимый вкус: " + this.state.value);
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          Выберите ваш любимый вкус:{" "}
          <select value={this.state.value} onChange={this.handleChange}>
            <option value="grapefruit">Грейпфрут</option>
            <option value="lime">Лайм</option>
            <option value="coconut">Кокос</option>
            <option value="mango">Манго</option>
          </select>
        </label>
        <input type="submit" value="Отправить" />
      </form>
    );
  }
}
```

# Lifecycle

```jsx
class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = { date: new Date() };
    //устанавливаем в state текущую дату(1)
  }
  componentDidMount() {
    this.timerID = setInterval(() => this.tick(), 1000);
  } //после отрисовки вызвать функцию tick(3)
  componentWillUnmount() {
    clearInterval(this.timerID);
    //перед тем как удалить отрисованный компонент, при смене state, сбрасываем таймер (5)
  }
  tick() {
    this.setState({ date: new Date() });
  } //функция tick в свою очередь меняет state, значит нужно отрисовать компонент заново (4)
  render() {
    return (
      <div>
        <h1>Привет, мир!</h1>
        <h2>Сейчас {this.state.date.toLocaleTimeString()}.</h2>
      </div>
    );
  }
}

ReactDOM.render(
  <Clock />,
  document.getElementById("root")
  //компонент отрисовывается, вызывая componentDidMount (2)
);
```

# props

Пропсы по умолчанию

```js
class Welcome extends React.Component {
  render() {
    return <h1>Hello {this.props.name}</h1>;
  }
}
Welcome.defaultProps = {
  name: "world",
};
```

Если мы вызовем компонент `<Welcome />` без аргументов, то будет `<h1>Hello World<h2>`

## render props

Компонент с рендер-проп берёт функцию, которая возвращает React-элемент, и вызывает её вместо реализации собственного рендера.

```jsx
<DataProvider render={(data) => <h1>Привет, {data.target}</h1>} />
```

приложение, которое отслеживает положение мыши

```jsx
class MouseTracker extends React.Component {
  constructor(props) {
    super(props);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.state = { x: 0, y: 0 };
  }
  handleMouseMove(event) {
    this.setState({
      x: event.clientX,
      y: event.clientY,
    });
  }
  render() {
    return (
      <div _style={{ height: "100vh" }} onMouseMove={this.handleMouseMove}>
        <h1>Перемещайте курсор мыши!</h1>
        <p>
          Текущее положение курсора мыши: ({this.state.x}, {this.state.y})
        </p>     {" "}
      </div>
    );
  }
}
```

```js
// Компонент <Mouse> инкапсулирует поведение, которое нам необходимо, но состояние не разделяется
class Mouse extends React.Component {
  constructor(props) {
    super(props);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.state = { x: 0, y: 0 };
  }
  handleMouseMove(event) {
    this.setState({
      x: event.clientX,
      y: event.clientY,
    });
  }
  render() {
    return (
      <div _style={{ height: "100vh" }} onMouseMove={this.handleMouseMove}>
        {/*но как можно отрендерить что-то, кроме <p>?*/}
        <p>
          Текущее положение курсора мыши: ({this.state.x}, {this.state.y})
        </p>
      </div>
    );
  }
}
//отделяем заголовок, а логику оставляем отдельно

class MouseTracker extends React.Component {
  render() {
    return (
      <>
        <h1>Перемещайте курсор мыши!</h1>
        <Mouse />
        {/*здесь будет только <p>*/}
      </>
    );
  }
}
```

Для начала вы можете отрендерить `<Cat>` внутри метода render компонента `<Mouse>` следующим образом:

```jsx
class Cat extends React.Component {
  render() {
    const mouse = this.props.mouse;
    //this.state {x: 0,y: 0}
    return (
      <img
        src="/cat.jpg"
        _style={{ position: "absolute", left: mouse.x, top: mouse.y }}
      />
    );
  }
}

class MouseWithCat extends React.Component {
  constructor(props) {
    super(props);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.state = { x: 0, y: 0 };
  }
  handleMouseMove(event) {
    this.setState({
      x: event.clientX,
      y: event.clientY,
    });
  }
  render() {
    return (
      <div _style={{ height: "100vh" }} onMouseMove={this.handleMouseMove}>
        <Cat mouse={this.state} />
      </div>
    );
  }
}
class MouseTracker extends React.Component {
  render() {
    return (
      <div>
        <h1>Перемещайте курсор мыши!</h1>
        <MouseWithCat />
      </div>
    );
  }
}
```

изображение кошки, которая двигается по экрану

```js
class Cat extends React.Component {
  render() {
    const mouse = this.props.mouse;
    return (
      <img
        src="/cat.jpg"
        _style={{ position: "absolute", left: mouse.x, top: mouse.y }}
      />
    );
  }
}
class Mouse extends React.Component {
  constructor(props) {
    super(props);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.state = { x: 0, y: 0 };
  }
  handleMouseMove(event) {
    this.setState({
      x: event.clientX,
      y: event.clientY,
    });
  }
  render() {
    return (
      <div _style={{ height: "100vh" }} onMouseMove={this.handleMouseMove}>
        {/*Вместо статического представления того, что рендерит <Mouse>, используем рендер-проп для динамического определения, что надо отрендерить*/}
        {this.props.render(this.state)} {/*в рендер функцию отправляется аргумент this.state */}
      </div>
    );
  }
}
class MouseTracker extends React.Component {
  render() {
    return (
      <div>
        <h1>Перемещайте курсор мыши!</h1>
        <Mouse
          render={(mouse) => (
            //проп рендер – это анонимная функция с аргументом mouse, здесь направится this.state в качестве mouse
            <Cat mouse={mouse} />
          )}
        />
      </div>
    );
  }
}
```

Минус данного решения = каждый раз при создании компонента, нужно создавать промежуточный компонент – типа MouseWithCat

```jsx
// Если вам действительно необходим HOC по некоторым причинам, вы можете просто создать обычный компонент с рендер-проп!
function withMouse(Component) {
  return class extends React.Component {
    render() {
      return (
        <Mouse render={mouse => (
          <Component {...this.props} mouse={mouse} />
)}/>);}}}

//Несмотря на то, что в вышеприведённых примерах мы используем render, мы можем также легко использовать проп children !
<Mouse children={mouse => (
  <p>Текущее положение курсора мыши: {mouse.x}, {mouse.y}</p>
)}/>

// children не обязательно именовать в списке «атрибутов» JSX-элемента. Вместо этого, его можно поместить его прямо внутрь элемента!
<Mouse>
  {mouse => (
    <p>Текущее положение курсора мыши: {mouse.x}, {mouse.y}</p>
  )}
</Mouse>

```

# ref

```jsx
import React from "react";
class CustomTextInput extends React.Component {
  constructor(props) {
    super(props);
    this.textInput = React.createRef(); // создадим реф в поле `textInput` для хранения DOM-элемента
    this.focusTextInput = this.focusTextInput.bind(this);
  }
  focusTextInput() {
    // Установим фокус на текстовое поле с помощью чистого DOM API
    // Примечание: обращаемся к "current", чтобы получить DOM-узел
    this.textInput.current.focus();
  }
  render() {
    // описываем, что мы хотим связать реф <input>
    // с `textInput` созданным в конструкторе
    return (
      <div>
        <input type="text" ref={this.textInput} /> присваиваем реф для инпута
        <input
          type="button"
          value="Фокус на текстовом поле"
          onClick={this.focusTextInput}
        />
      </div>
    );
  }
}
```

# state

Неправильно
this.state.comment = 'Привет';
Правильно
this.setState({comment: 'Привет'});

!!!Конструктор — это единственное место, где вы можете присвоить значение this.state напрямую
!!!При обновлении состояния компонент рендерится заново
!!!Хранить состояние нужно на самом верху иерархии компонентов

```jsx
// изменение состояния с помощью пропсов
// Неправильно
this.setState({
  counter: this.state.counter + this.props.increment,
});
// Правильно
this.setState((state, props) => ({
  counter: state.counter + props.increment,
}));
// Правильно
this.setState(function (state, props) {
  return {
    counter: state.counter + props.increment,
  };
});
```

Компонент может передать своё состояние вниз по дереву в виде пропсов дочерних компонентов:

```jsx
class Clock extends React.Component {
  //изменен из примера с часами

  render() {
    return (
      <div>
        <h1>Hello, world!</h1>
        FormattedDate date={this.state.date} /> {/*передача текущего значения пропсов*/}
      </div>
    );
  }
}
ReactDOM.render(<Clock />, document.getElementById("root"));
```

## Подъем состояния

```js
class Calculator extends React.Component {
  constructor(props) {
    super(props);
    this.handleCelsiusChange = this.handleCelsiusChange.bind(this); //обработчик для поля Цельсия, который меняет состояние
    this.handleFahrenheitChange = this.handleFahrenheitChange.bind(this); //обработчик для поля Фаренгейт
    this.state = { temperature: "", scale: "c" };
  } //в состоянии поле текущей температуры и шкалы для указания того, что меняется}

  handleCelsiusChange(temperature) {
    this.setState({ scale: "c", temperature });
  } //если ввод происходит здесь то состояние меняется на scale: ‘c’ и устанавливает текущее значение температуры в Цельсия
  handleFahrenheitChange(temperature) {
    this.setState({ scale: "f", temperature });
  }
  //если ввод происходит здесь то состояние меняется на scale: ‘f’ и устанавливает значение температуры в фаренгейтах
  render() {
    const scale = this.state.scale; //присваиваем переменной scale значение из состояния
    const temperature = this.state.temperature; //присваиваем переменной temperature значение из состояния
    const celsius =
      scale === "f" ? tryConvert(temperature, toCelsius) : temperature; //рендерится два компонента со значением scale === f и со значением scale === c, в зависимости от этого происходит конвертация
    const fahrenheit =
      scale === "c" ? tryConvert(temperature, toFahrenheit) : temperature;
    return (
      <div>
        <TemperatureInput
          scale="c"
          temperature={celsius}
          onTemperatureChange={this.handleCelsiusChange}
        />

        <TemperatureInput
          scale="f"
          temperature={fahrenheit}
          onTemperatureChange={this.handleFahrenheitChange}
        />
        <BoilingVerdict celsius={parseFloat(celsius)} />
      </div>
    );
  }
}

// <TemperatureInput scale="c“ temperature={celsius} onTemperatureChange={this.handleCelsiusChange}/>
// <TemperatureInput scale="f“ temperature={fahrenheit} onTemperatureChange={this.handleFahrenheitChange}/>
class TemperatureInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
  }
  handleChange(e) {
    this.props.onTemperatureChange(e.target.value);
  }
  render() {
    const temperature = this.props.temperature;
    const scale = this.props.scale;
    return (
      <fieldset>
        <legend>Введите градусы по шкале {scaleNames[scale]}:</legend>
        <input value={temperature} onChange={this.handleChange} />
      </fieldset>
    );
  }
}

function toCelsius(fahrenheit) {
  return ((fahrenheit - 32) * 5) / 9;
}
function toFahrenheit(celsius) {
  return (celsius * 9) / 5 + 32;
}
function tryConvert(temperature, convert) {
  //функция для обработки случая ввода не численных значений и округления результата
  const input = parseFloat(temperature);
  if (Number.isNaN(input)) {
    return "";
  }
  const output = convert(input); // задействуем коллбек
  const rounded = Math.round(output * 1000) / 1000;
  return rounded.toString();
}
function BoilingVerdict(props) {
  //компонент для определения закипания
  if (props.celsius >= 100) {
    return <p>Вода закипит.</p>;
  }
  return <p>Вода не закипит.</p>;
}
```

# Styling

## Inline стили

```js
class ToDoApp extends React.Component {
  render() {
    return (
      <div
        _style={{
          backgroundColor: "#44014C",
          width: "300px",
          minHeight: "200px",
        }}
      >
        <h2
          _style={{ padding: "10px 20px", textAlign: "center", color: "white" }}
        >
          ToDo
        </h2>
        <div>
          <Input onChange={this.handleChange} />
          <p>{this.state.error}</p>
          <ToDoList value={this.state.display} /> 
        </div> 
      </div>
    );
  }
}
```

## Объект стилей

```jsx
import React, { Component } from "react";
import "./App.css";
class App extends Component {
  render() {
    const divStyle = {
      textAlign: "center", //можно и ‘text align’ : ‘center’ но в консоли вылезут ошибки по исправлению на camelCase
    };
    return (
      <div _style={divStyle}>
        <h1 _style={{ color: "blue", fontSize: "20px" }}>Hello</h1> 
      </div>
    );
  }
}
export default App;
```

## Объект стилей в раздельных файлах

```js
//styles.js
const TodoComponent = {
  width: "300px",
  margin: "30px auto",
  backgroundColor: "#44014C",
  minHeight: "200px",
  boxSizing: "border-box"
}

const Header = {
  padding: "10px 20px",
  textAlign: "center",
  color: "white",
  fontSize: "22px"
}

const ErrorMessage = {
  color: "white",
  fontSize: "13px"
}

const styles = {
  //объединим в один объект для экспорта, в противном случае нужно будет все экспортировать отдельно
  TodoComponent: TodoComponent,
  Header: Header,
  ErrorMessage: ErrorMessage
}

export styles
```

```jsx
// app.js;
// Import the styles
import { styles } from "./styles";
class ToDoApp extends React.Component {
  // ...
  render() {
    return (
      <div _style={styles.TodoComponent}>
        <h2 _style={styles.Header}>ToDo</h2> 
        <div>
          <Input onChange={this.handleChange} />   
          <p _style={styles.ErrorMessage}>{this.state.error}</p>
          <ToDoList value={this.state.display} /> 
        </div>
      </div>
    );
  }
}
```
