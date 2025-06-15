# jsx

- React Element – это объектное представление каждого компонента react
  {type: «html-tag», prop: any, children: ReactNode[]}, они иммутабельные.
- Есть два типа элемента React DOM Element – то, что можно представить в виде html-тега и React Element Component – обертка для совокупности React DOM Element-ов, которые в свою очередь превращаются в DOM дерево в последствии рендера

```jsx
import React, { Component } from "react";
import "./App.css";
class App extends Component {
  render() {
    /*обычный метод рендеринга:
     return (
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
