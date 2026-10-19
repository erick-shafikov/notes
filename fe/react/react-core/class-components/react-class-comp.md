!!!Р¶РёР·РЅРµРЅРЅС‹Р№ С†РёРєР» РїРµСЂРµРЅРµСЃС‚Рё РІ СЌС‚Сѓ РїР°РїРєСѓ

# РљРѕРјРїРѕР·РёС†РёСЏ

РїСЂРѕРї children РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РґР»СЏ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ, РІ РєРѕС‚РѕСЂС‹С… СЃРѕРґРµСЂР¶РёРјРѕРµ РЅРµРѕРїСЂРµРґРµР»РµРЅРЅРѕ РґРѕ СЂРµРЅРґРµСЂРёРЅРіР°

```tsx
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
    alert(`Р”РѕР±СЂРѕ РїРѕР¶Р°Р»РѕРІР°С‚СЊ РЅР° Р±РѕСЂС‚, ${this.state.login}!`);
  }
  render() {
    return (
      <Dialog
        title="РџСЂРѕРіСЂР°РјРјР° РёСЃСЃР»РµРґРѕРІР°РЅРёСЏ РњР°СЂСЃР°"
        message="РљР°Рє Рє РІР°Рј РѕР±СЂР°С‰Р°С‚СЊСЃСЏ?"
      >
        <input value={this.state.login} onChange={this.handleChange} />
        <button onClick={this.handleSignUp}>РљРЅРѕРїРєР°</button>
      </Dialog>
    );
  }
}
```

# conditional rendering

РћРЅ Р±СѓРґРµС‚ СЂРµРЅРґРµСЂРёС‚СЊ Р»РёР±Рѕ <LoginButton />, Р»РёР±Рѕ <LogoutButton /> РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ С‚РµРєСѓС‰РµРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ. Рђ РµС‰С‘ РѕРЅ Р±СѓРґРµС‚ РІСЃРµРіРґР° СЂРµРЅРґРµСЂРёС‚СЊ <Greeting /> РёР· РїСЂРµРґС‹РґСѓС‰РµРіРѕ РїСЂРёРјРµСЂР°.

```tsx
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

Р·РЅР°С‡РµРЅРёРµ РїСЂРѕРїСЃР° theme РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ РїРµСЂРµРґР°РЅРѕ РІРЅРёР· РёР· T

```tsx
class Example extends React.Component {
  render() {
    return <Toolbar theme="dark" />;
  }
}
//РїРµСЂРµРµРґР°РµС‚ РїСЂРѕРї theme РІ toolbar

function Toolbar(props) {
  return (
    <div>
      <ThemedButton theme={props.theme} />
    </div>
  );
}
//РїРµСЂРµРµРґР°РµС‚ РїСЂРѕРї theme РІ ThemeButton
class ThemedButton extends React.Component {
  render() {
    return <Button theme={this.props.theme} />;
  }
} //РїРµСЂРµРµРґР°РµС‚ РїСЂРѕРї theme РІ Button
class Button extends React.Component {
  render() {
    return <button>{this.props.theme}</button>;
  }
}
```

СЃ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёРµРј РєРѕРЅС‚РµРєСЃС‚Р°

```tsx
const ThemeContext = React.createContext("light");
//СЃРѕР·РґР°РµРј РєРѕРЅС‚РµРєСЃС‚ СЃ РїРѕРјРѕС‰СЊСЋ createContext
class Example extends React.Component {
  render() {
    return (
      //РїРµСЂРµРґР°РµРј РїСЂРѕР±СЂР°СЃС‹РІР°РµРјРѕРµ Р·РЅР°С‡РµРЅРёРµ
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
  //С„РёРєСЃРёСЂСѓРµРј РјРµСЃС‚РѕРЅР°С…РѕР¶РґРµРЅРёРµ
  render() {
    return <button>{this.context}</button>; //РёСЃРїРѕР»СЊР·СѓРµРј
  }
}
```

```tsx
const ThemeContext1 = React.createContext({ color: "green" }); //(1.1)СЃРѕР·РґР°РµРј РѕР±СЉРµРєС‚ ThemeContext
const ThemeContext2 = React.createContext(); //СЃРѕР·РґР°РµРј РІС‚РѕСЂРѕР№ РѕР±СЉРµРєС‚ РєРѕРЅС‚РµРєСЃС‚Р° (1.2)
class Example extends React.Component {
  render() {
    return (
      //РѕС‚РїСЂР°РІР»СЏРµРј РїРµСЂРІС‹Р№ (2.1)
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
    //РѕС‚РїСЂР°РІР»СЏРµРј РІС‚РѕСЂРѕР№ (2.2)
    // В  В  В  context: {this.context.color}
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
  static contextType = ThemeContext2; //(3) РёРґРµС‚ РІ РїР°СЂРµ СЃ (4)
  render() {
    return (
      <button _style={{ backgroundColor: `${this.context}` }}>
        {this.context}
      </button>
    ); //(5) РґРѕСЃС‚Р°РµРј Р·РЅР°С‡РµРЅРёРµ РєРѕРЅС‚РµРєСЃС‚Р°
  }
}
```

# Events handler

```tsx
// HTML: <button onclick="activateLasers()"> РђРєС‚РёРІРёСЂРѕРІР°С‚СЊ Р»Р°Р·РµСЂС‹ </button>
// JSX: <button onClick={activateLasers}> РђРєС‚РёРІРёСЂРѕРІР°С‚СЊ Р»Р°Р·РµСЂС‹ </button>

// РџСЂРёРјРµСЂ РїСЂРёРІСЏР·РєРё СЃРѕР±С‹С‚РёСЏ С‡РµСЂРµР· РєРѕРЅСЃС‚СЂСѓРєС‚РѕСЂ РІ РєР»Р°СЃСЃРѕРІРѕРј РєРѕРјРїРѕРЅРµРЅС‚Рµ

class Toggle extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isToggleOn: true };
    this.handleClick = this.handleClick.bind(this); //РЅРµРѕР±С…РѕРґРёРјРѕ РґР»СЏ СЂР°Р±РѕС‚С‹ РІ РєРѕР»Р±СЌРєРµ
  }
  handleClick() {
    this.setState((prevState) => ({
      isToggleOn: !prevState.isToggle, //РјРµРЅСЏРµРј РЅР° РїСЂРѕС‚РёРІРѕРїРѕР»РѕР¶РЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ
    }));
  }
  render() {
    return (
      <button onClick={this.handleClick}>
        {/*РїСЂРёСЂР°РІРЅРёРІР°РЅРёРµ Р±РµР· Р°СЂРіСѓРјРµРЅС‚РѕРІ*/}
        {this.state.isToggleOn ? "ON" : "OFF"}
        {/*РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ СЃРѕСЃС‚РѕСЏРЅРёСЏ РѕС‚РѕР±СЂР°Р¶Р°С‚СЊ ON РёР»Рё OFF*/}
      </button>
    );
  }
}
```

```tsx
class LoggingButton extends React.Component {
  // РўР°РєРѕР№ СЃРёРЅС‚Р°РєСЃРёСЃ РіР°СЂР°РЅС‚РёСЂСѓРµС‚, С‡С‚Рѕ `this` РїСЂРёРІСЏР·Р°РЅ Рє handleClick.
  // РџСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ: СЌС‚Рѕ СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°Р»СЊРЅС‹Р№ СЃРёРЅС‚Р°РєСЃРёСЃ
  handleClick = () => {
    //РјРµРЅСЏРµРј РѕР±С‹С‡РЅРѕРµ РїРѕР»Рµ РЅР° СЃС‚СЂРµР»РѕС‡РЅСѓСЋ С„СѓРЅРєС†РёСЋ
    console.log("Р·РЅР°С‡РµРЅРёРµ this:", this);
  };
  render() {
    return <button onClick={this.handleClick}>РќР°Р¶РјРё РЅР° РјРµРЅСЏ</button>;
  }
}

// СЃ РїРѕРјРѕС‰СЊСЋ РєРѕР»Р»Р±РµРєР° РІ РѕР±СЂР°Р±РѕС‚С‡РёРєРµ СЃРѕР±С‹С‚РёСЏ
class LoggingButton extends React.Component {
  handleClick() {
    console.log("Р·РЅР°С‡РµРЅРёРµ this:", this);
  }
  render() {
    // РўР°РєРѕР№ СЃРёРЅС‚Р°РєСЃРёСЃ РіР°СЂР°РЅС‚РёСЂСѓРµС‚, С‡С‚Рѕ `this` РїСЂРёРІСЏР·Р°РЅ Рє handleClick.
    return <button onClick={() => this.handleClick()}>РќР°Р¶РјРё РЅР° РјРµРЅСЏ</button>;
  }
}
```

## РџРµСЂРµРґР°С‡Р° Р°СЂРіСѓРјРµРЅС‚РѕРІ РІ РѕР±СЂР°Р±РѕС‚С‡РёРєРё СЃРѕР±С‹С‚РёР№

```tsx

<button onClick={(e) => this.deleteRow(id, e)}>РЈРґР°Р»РёС‚СЊ СЃС‚СЂРѕРєСѓ</button>
<button onClick={this.deleteRow.bind(this, id)}>РЈРґР°Р»РёС‚СЊ СЃС‚СЂРѕРєСѓ</button>
```

Р’ РѕР±РѕРёС… СЃР»СѓС‡Р°СЏС… Р°СЂРіСѓРјРµРЅС‚ e, РїСЂРµРґСЃС‚Р°РІР»СЏСЋС‰РёР№ СЃРѕР±С‹С‚РёРµ React, Р±СѓРґРµС‚ РїРµСЂРµРґР°РЅ РєР°Рє РІС‚РѕСЂРѕР№ Р°СЂРіСѓРјРµРЅС‚ РїРѕСЃР»Рµ РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂР°. РСЃРїРѕР»СЊР·СѓСЏ СЃС‚СЂРµР»РѕС‡РЅСѓСЋ С„СѓРЅРєС†РёСЋ, РЅРµРѕР±С…РѕРґРёРјРѕ РїРµСЂРµРґР°РІР°С‚СЊ Р°СЂРіСѓРјРµРЅС‚ СЏРІРЅРѕ, РЅРѕ СЃ bind Р»СЋР±С‹Рµ РїРѕСЃР»РµРґСѓСЋС‰РёРµ Р°СЂРіСѓРјРµРЅС‚С‹ РїРµСЂРµРґР°СЋС‚СЃСЏ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё.

!!!РІ React РЅРµР»СЊР·СЏ РїСЂРµРґРѕС‚РІСЂР°С‚РёС‚СЊ РѕР±СЂР°Р±РѕС‚С‡РёРє СЃРѕР±С‹С‚РёСЏ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ, РІРµСЂРЅСѓРІ false. РќСѓР¶РЅРѕ СЏРІРЅРѕ РІС‹Р·РІР°С‚СЊ preventDefault

# forms

!!!РћР±РЅРѕРІР»РµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ С„РѕСЂРј С‚РѕР»СЊРєРѕ С‡РµСЂРµР· setstate
!!!СЃР±СЂРѕСЃ РґРµР№СЃС‚РІРёСЏ С‚РѕР»СЊРєРѕ С‡РµСЂРµР· event.preventDefault()

```tsx
class NameForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = { value: "" }; //СЃРѕСЃС‚РѕСЏРЅРёРµ
    this.handleChange = this.handleChange.bind(this); //РёР·РјРµРЅРµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ РїСЂРё РІРІРѕРґРµ
    this.handleSubmit = this.handleSubmit.bind(this); //РёР·РјРµРЅРµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ РїСЂРё submit
  }
  handleChange(event) {
    this.setState({ value: event.target.value }); //РїРµСЂРµР·Р°РїРёСЃСЊ СЃРѕСЃС‚РѕСЏРЅРёСЏ РїСЂРё РєР°Р¶РґРѕРј РІРІРѕРґРµ
  }
  handleSubmit(event) {
    alert("РћС‚РїСЂР°РІР»РµРЅРЅРѕРµ РёРјСЏ: " + this.state.value); //РїСЂРё submit РІРІС‹РІРµСЃС‚Рё Р·РЅР°С‡РµРЅРёРµ РїРѕР»СЏ
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          РРјСЏ:
          <input
            type="text"
            value={this.state.value}
            onChange={this.handleChange}
          />
        </label>
        <input type="submit" value="РћС‚РїСЂР°РІРёС‚СЊ" />
      </form>
    );
  }
}
```

РўРµРі `<textarea>`

```tsx
class EssayForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      value: "Р‘СѓРґСЊС‚Рµ Р»СЋР±РµР·РЅС‹, РЅР°РїРёС€РёС‚Рµ СЃРѕС‡РёРЅРµРЅРёРµ Рѕ РІР°С€РµРј Р»СЋР±РёРјРѕРј DOM-СЌР»РµРјРµРЅС‚Рµ.",
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }
  handleChange(event) {
    this.setState({ value: event.target.value });
  }
  handleSubmit(event) {
    alert("РЎРѕС‡РёРЅРµРЅРёРµ РѕС‚РїСЂР°РІР»РµРЅРѕ: " + this.state.value);
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          РЎРѕС‡РёРЅРµРЅРёРµ:
          <textarea value={this.state.value} onChange={this.handleChange} />
        </label>
        <input type="submit" value="РћС‚РїСЂР°РІРёС‚СЊ" />
      </form>
    );
  }
}
```

## Р¤РѕСЂРјС‹. `<select>`

```tsx
class FlavorForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = { value: "coconut" }; //РІС‹Р±СЂР°РЅ РљРѕРєРѕСЃ
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }
  handleChange(event) {
    this.setState({ value: event.target.value });
  }
  handleSubmit(event) {
    //РїСЂРё submit вЂ“ alert РІС‹Р±СЂР°РЅРЅРѕРіРѕ РїСѓРЅРєС‚Р° СЃРѕС…СЂР°РЅРµРЅРЅРѕРіРѕ РІ state
    alert("Р’Р°С€ Р»СЋР±РёРјС‹Р№ РІРєСѓСЃ: " + this.state.value);
    event.preventDefault();
  }
  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <label>
          Р’С‹Р±РµСЂРёС‚Рµ РІР°С€ Р»СЋР±РёРјС‹Р№ РІРєСѓСЃ:{" "}
          <select value={this.state.value} onChange={this.handleChange}>
            <option value="grapefruit">Р“СЂРµР№РїС„СЂСѓС‚</option>
            <option value="lime">Р›Р°Р№Рј</option>
            <option value="coconut">РљРѕРєРѕСЃ</option>
            <option value="mango">РњР°РЅРіРѕ</option>
          </select>
        </label>
        <input type="submit" value="РћС‚РїСЂР°РІРёС‚СЊ" />
      </form>
    );
  }
}
```

# Lifecycle

```tsx
class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = { date: new Date() };
    //СѓСЃС‚Р°РЅР°РІР»РёРІР°РµРј РІ state С‚РµРєСѓС‰СѓСЋ РґР°С‚Сѓ(1)
  }
  componentDidMount() {
    this.timerID = setInterval(() => this.tick(), 1000);
  } //РїРѕСЃР»Рµ РѕС‚СЂРёСЃРѕРІРєРё РІС‹Р·РІР°С‚СЊ С„СѓРЅРєС†РёСЋ tick(3)
  componentWillUnmount() {
    clearInterval(this.timerID);
    //РїРµСЂРµРґ С‚РµРј РєР°Рє СѓРґР°Р»РёС‚СЊ РѕС‚СЂРёСЃРѕРІР°РЅРЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚, РїСЂРё СЃРјРµРЅРµ state, СЃР±СЂР°СЃС‹РІР°РµРј С‚Р°Р№РјРµСЂ (5)
  }
  tick() {
    this.setState({ date: new Date() });
  } //С„СѓРЅРєС†РёСЏ tick РІ СЃРІРѕСЋ РѕС‡РµСЂРµРґСЊ РјРµРЅСЏРµС‚ state, Р·РЅР°С‡РёС‚ РЅСѓР¶РЅРѕ РѕС‚СЂРёСЃРѕРІР°С‚СЊ РєРѕРјРїРѕРЅРµРЅС‚ Р·Р°РЅРѕРІРѕ (4)
  render() {
    return (
      <div>
        <h1>РџСЂРёРІРµС‚, РјРёСЂ!</h1>
        <h2>РЎРµР№С‡Р°СЃ {this.state.date.toLocaleTimeString()}.</h2>
      </div>
    );
  }
}

ReactDOM.render(
  <Clock />,
  document.getElementById("root")
  //РєРѕРјРїРѕРЅРµРЅС‚ РѕС‚СЂРёСЃРѕРІС‹РІР°РµС‚СЃСЏ, РІС‹Р·С‹РІР°СЏ componentDidMount (2)
);
```

# props

РџСЂРѕРїСЃС‹ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ

```tsx
class Welcome extends React.Component {
  render() {
    return <h1>Hello {this.props.name}</h1>;
  }
}
Welcome.defaultProps = {
  name: "world",
};
```

Р•СЃР»Рё РјС‹ РІС‹Р·РѕРІРµРј РєРѕРјРїРѕРЅРµРЅС‚ `<Welcome />` Р±РµР· Р°СЂРіСѓРјРµРЅС‚РѕРІ, С‚Рѕ Р±СѓРґРµС‚ `<h1>Hello World<h2>`

## render props

РљРѕРјРїРѕРЅРµРЅС‚ СЃ СЂРµРЅРґРµСЂ-РїСЂРѕРї Р±РµСЂС‘С‚ С„СѓРЅРєС†РёСЋ, РєРѕС‚РѕСЂР°СЏ РІРѕР·РІСЂР°С‰Р°РµС‚ React-СЌР»РµРјРµРЅС‚, Рё РІС‹Р·С‹РІР°РµС‚ РµС‘ РІРјРµСЃС‚Рѕ СЂРµР°Р»РёР·Р°С†РёРё СЃРѕР±СЃС‚РІРµРЅРЅРѕРіРѕ СЂРµРЅРґРµСЂР°.

```tsx
<DataProvider render={(data) => <h1>РџСЂРёРІРµС‚, {data.target}</h1>} />
```

РїСЂРёР»РѕР¶РµРЅРёРµ, РєРѕС‚РѕСЂРѕРµ РѕС‚СЃР»РµР¶РёРІР°РµС‚ РїРѕР»РѕР¶РµРЅРёРµ РјС‹С€Рё

```tsx
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
        <h1>РџРµСЂРµРјРµС‰Р°Р№С‚Рµ РєСѓСЂСЃРѕСЂ РјС‹С€Рё!</h1>
        <p>
          РўРµРєСѓС‰РµРµ РїРѕР»РѕР¶РµРЅРёРµ РєСѓСЂСЃРѕСЂР° РјС‹С€Рё: ({this.state.x}, {this.state.y})
        </p>В  В  В {" "}
      </div>
    );
  }
}
```

```tsx
// РљРѕРјРїРѕРЅРµРЅС‚ <Mouse> РёРЅРєР°РїСЃСѓР»РёСЂСѓРµС‚ РїРѕРІРµРґРµРЅРёРµ, РєРѕС‚РѕСЂРѕРµ РЅР°Рј РЅРµРѕР±С…РѕРґРёРјРѕ, РЅРѕ СЃРѕСЃС‚РѕСЏРЅРёРµ РЅРµ СЂР°Р·РґРµР»СЏРµС‚СЃСЏ
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
        {/*РЅРѕ РєР°Рє РјРѕР¶РЅРѕ РѕС‚СЂРµРЅРґРµСЂРёС‚СЊ С‡С‚Рѕ-С‚Рѕ, РєСЂРѕРјРµ <p>?*/}
        <p>
          РўРµРєСѓС‰РµРµ РїРѕР»РѕР¶РµРЅРёРµ РєСѓСЂСЃРѕСЂР° РјС‹С€Рё: ({this.state.x}, {this.state.y})
        </p>
      </div>
    );
  }
}
//РѕС‚РґРµР»СЏРµРј Р·Р°РіРѕР»РѕРІРѕРє, Р° Р»РѕРіРёРєСѓ РѕСЃС‚Р°РІР»СЏРµРј РѕС‚РґРµР»СЊРЅРѕ

class MouseTracker extends React.Component {
  render() {
    return (
      <>
        <h1>РџРµСЂРµРјРµС‰Р°Р№С‚Рµ РєСѓСЂСЃРѕСЂ РјС‹С€Рё!</h1>
        <Mouse />
        {/*Р·РґРµСЃСЊ Р±СѓРґРµС‚ С‚РѕР»СЊРєРѕ <p>*/}
      </>
    );
  }
}
```

Р”Р»СЏ РЅР°С‡Р°Р»Р° РІС‹ РјРѕР¶РµС‚Рµ РѕС‚СЂРµРЅРґРµСЂРёС‚СЊ `<Cat>` РІРЅСѓС‚СЂРё РјРµС‚РѕРґР° render РєРѕРјРїРѕРЅРµРЅС‚Р° `<Mouse>` СЃР»РµРґСѓСЋС‰РёРј РѕР±СЂР°Р·РѕРј:

```tsx
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
        <h1>РџРµСЂРµРјРµС‰Р°Р№С‚Рµ РєСѓСЂСЃРѕСЂ РјС‹С€Рё!</h1>
        <MouseWithCat />
      </div>
    );
  }
}
```

РёР·РѕР±СЂР°Р¶РµРЅРёРµ РєРѕС€РєРё, РєРѕС‚РѕСЂР°СЏ РґРІРёРіР°РµС‚СЃСЏ РїРѕ СЌРєСЂР°РЅСѓ

```tsx
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
        {/*Р’РјРµСЃС‚Рѕ СЃС‚Р°С‚РёС‡РµСЃРєРѕРіРѕ РїСЂРµРґСЃС‚Р°РІР»РµРЅРёСЏ С‚РѕРіРѕ, С‡С‚Рѕ СЂРµРЅРґРµСЂРёС‚ <Mouse>, РёСЃРїРѕР»СЊР·СѓРµРј СЂРµРЅРґРµСЂ-РїСЂРѕРї РґР»СЏ РґРёРЅР°РјРёС‡РµСЃРєРѕРіРѕ РѕРїСЂРµРґРµР»РµРЅРёСЏ, С‡С‚Рѕ РЅР°РґРѕ РѕС‚СЂРµРЅРґРµСЂРёС‚СЊ*/}
        {this.props.render(this.state)}В {/*РІ СЂРµРЅРґРµСЂ С„СѓРЅРєС†РёСЋ РѕС‚РїСЂР°РІР»СЏРµС‚СЃСЏ Р°СЂРіСѓРјРµРЅС‚ this.state */}
      </div>
    );
  }
}
class MouseTracker extends React.Component {
  render() {
    return (
      <div>
        <h1>РџРµСЂРµРјРµС‰Р°Р№С‚Рµ РєСѓСЂСЃРѕСЂ РјС‹С€Рё!</h1>
        <Mouse
          render={(mouse) => (
            //РїСЂРѕРї СЂРµРЅРґРµСЂ вЂ“ СЌС‚Рѕ Р°РЅРѕРЅРёРјРЅР°СЏ С„СѓРЅРєС†РёСЏ СЃ Р°СЂРіСѓРјРµРЅС‚РѕРј mouse, Р·РґРµСЃСЊ РЅР°РїСЂР°РІРёС‚СЃСЏ this.state РІ РєР°С‡РµСЃС‚РІРµ mouse
            <Cat mouse={mouse} />
          )}
        />
      </div>
    );
  }
}
```

РњРёРЅСѓСЃ РґР°РЅРЅРѕРіРѕ СЂРµС€РµРЅРёСЏ = РєР°Р¶РґС‹Р№ СЂР°Р· РїСЂРё СЃРѕР·РґР°РЅРёРё РєРѕРјРїРѕРЅРµРЅС‚Р°, РЅСѓР¶РЅРѕ СЃРѕР·РґР°РІР°С‚СЊ РїСЂРѕРјРµР¶СѓС‚РѕС‡РЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚ вЂ“ С‚РёРїР° MouseWithCat

```tsx
// Р•СЃР»Рё РІР°Рј РґРµР№СЃС‚РІРёС‚РµР»СЊРЅРѕ РЅРµРѕР±С…РѕРґРёРј HOC РїРѕ РЅРµРєРѕС‚РѕСЂС‹Рј РїСЂРёС‡РёРЅР°Рј, РІС‹ РјРѕР¶РµС‚Рµ РїСЂРѕСЃС‚Рѕ СЃРѕР·РґР°С‚СЊ РѕР±С‹С‡РЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚ СЃ СЂРµРЅРґРµСЂ-РїСЂРѕРї!
function withMouse(Component) {
В  return class extends React.Component {
В  В  render() {
В  В  В  return (
В  В  В  В  <Mouse render={mouse => (
В  В  В  В  В  <Component {...this.props} mouse={mouse} />
)}/>);}}}

//РќРµСЃРјРѕС‚СЂСЏ РЅР° С‚Рѕ, С‡С‚Рѕ РІ РІС‹С€РµРїСЂРёРІРµРґС‘РЅРЅС‹С… РїСЂРёРјРµСЂР°С… РјС‹ РёСЃРїРѕР»СЊР·СѓРµРј render, РјС‹ РјРѕР¶РµРј С‚Р°РєР¶Рµ Р»РµРіРєРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РїСЂРѕРї children !
<Mouse children={mouse => (
В  <p>РўРµРєСѓС‰РµРµ РїРѕР»РѕР¶РµРЅРёРµ РєСѓСЂСЃРѕСЂР° РјС‹С€Рё: {mouse.x}, {mouse.y}</p>
)}/>

// children РЅРµ РѕР±СЏР·Р°С‚РµР»СЊРЅРѕ РёРјРµРЅРѕРІР°С‚СЊ РІ СЃРїРёСЃРєРµ В«Р°С‚СЂРёР±СѓС‚РѕРІВ» JSX-СЌР»РµРјРµРЅС‚Р°. Р’РјРµСЃС‚Рѕ СЌС‚РѕРіРѕ, РµРіРѕ РјРѕР¶РЅРѕ РїРѕРјРµСЃС‚РёС‚СЊ РµРіРѕ РїСЂСЏРјРѕ РІРЅСѓС‚СЂСЊ СЌР»РµРјРµРЅС‚Р°!
<Mouse>
В  {mouse => (
В  В  <p>РўРµРєСѓС‰РµРµ РїРѕР»РѕР¶РµРЅРёРµ РєСѓСЂСЃРѕСЂР° РјС‹С€Рё: {mouse.x}, {mouse.y}</p>
В  )}
</Mouse>

```

# ref

```tsx
import React from "react";
class CustomTextInput extends React.Component {
  constructor(props) {
    super(props);
    this.textInput = React.createRef(); // СЃРѕР·РґР°РґРёРј СЂРµС„ РІ РїРѕР»Рµ `textInput` РґР»СЏ С…СЂР°РЅРµРЅРёСЏ DOM-СЌР»РµРјРµРЅС‚Р°
    this.focusTextInput = this.focusTextInput.bind(this);
  }
  focusTextInput() {
    // РЈСЃС‚Р°РЅРѕРІРёРј С„РѕРєСѓСЃ РЅР° С‚РµРєСЃС‚РѕРІРѕРµ РїРѕР»Рµ СЃ РїРѕРјРѕС‰СЊСЋ С‡РёСЃС‚РѕРіРѕ DOM API
    // РџСЂРёРјРµС‡Р°РЅРёРµ: РѕР±СЂР°С‰Р°РµРјСЃСЏ Рє "current", С‡С‚РѕР±С‹ РїРѕР»СѓС‡РёС‚СЊ DOM-СѓР·РµР»
    this.textInput.current.focus();
  }
  render() {
    // РѕРїРёСЃС‹РІР°РµРј, С‡С‚Рѕ РјС‹ С…РѕС‚РёРј СЃРІСЏР·Р°С‚СЊ СЂРµС„ <input>
    // СЃ `textInput` СЃРѕР·РґР°РЅРЅС‹Рј РІ РєРѕРЅСЃС‚СЂСѓРєС‚РѕСЂРµ
    return (
      <div>
        <input type="text" ref={this.textInput} /> РїСЂРёСЃРІР°РёРІР°РµРј СЂРµС„ РґР»СЏ РёРЅРїСѓС‚Р°
        <input
          type="button"
          value="Р¤РѕРєСѓСЃ РЅР° С‚РµРєСЃС‚РѕРІРѕРј РїРѕР»Рµ"
          onClick={this.focusTextInput}
        />
      </div>
    );
  }
}
```

# state

РќРµРїСЂР°РІРёР»СЊРЅРѕ
this.state.comment = 'РџСЂРёРІРµС‚';
РџСЂР°РІРёР»СЊРЅРѕ
this.setState({comment: 'РџСЂРёРІРµС‚'});

!!!РљРѕРЅСЃС‚СЂСѓРєС‚РѕСЂ вЂ” СЌС‚Рѕ РµРґРёРЅСЃС‚РІРµРЅРЅРѕРµ РјРµСЃС‚Рѕ, РіРґРµ РІС‹ РјРѕР¶РµС‚Рµ РїСЂРёСЃРІРѕРёС‚СЊ Р·РЅР°С‡РµРЅРёРµ this.state РЅР°РїСЂСЏРјСѓСЋ
!!!РџСЂРё РѕР±РЅРѕРІР»РµРЅРёРё СЃРѕСЃС‚РѕСЏРЅРёСЏ РєРѕРјРїРѕРЅРµРЅС‚ СЂРµРЅРґРµСЂРёС‚СЃСЏ Р·Р°РЅРѕРІРѕ
!!!РҐСЂР°РЅРёС‚СЊ СЃРѕСЃС‚РѕСЏРЅРёРµ РЅСѓР¶РЅРѕ РЅР° СЃР°РјРѕРј РІРµСЂС…Сѓ РёРµСЂР°СЂС…РёРё РєРѕРјРїРѕРЅРµРЅС‚РѕРІ

```tsx
// РёР·РјРµРЅРµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ СЃ РїРѕРјРѕС‰СЊСЋ РїСЂРѕРїСЃРѕРІ
// РќРµРїСЂР°РІРёР»СЊРЅРѕ
this.setState({
  counter: this.state.counter + this.props.increment,
});
// РџСЂР°РІРёР»СЊРЅРѕ
this.setState((state, props) => ({
  counter: state.counter + props.increment,
}));
// РџСЂР°РІРёР»СЊРЅРѕ
this.setState(function (state, props) {
  return {
    counter: state.counter + props.increment,
  };
});
```

РљРѕРјРїРѕРЅРµРЅС‚ РјРѕР¶РµС‚ РїРµСЂРµРґР°С‚СЊ СЃРІРѕС‘ СЃРѕСЃС‚РѕСЏРЅРёРµ РІРЅРёР· РїРѕ РґРµСЂРµРІСѓ РІ РІРёРґРµ РїСЂРѕРїСЃРѕРІ РґРѕС‡РµСЂРЅРёС… РєРѕРјРїРѕРЅРµРЅС‚РѕРІ:

```tsx
class Clock extends React.Component {
  //РёР·РјРµРЅРµРЅ РёР· РїСЂРёРјРµСЂР° СЃ С‡Р°СЃР°РјРё

  render() {
    return (
      <div>
        <h1>Hello, world!</h1>
        FormattedDate date={this.state.date} /> {/*РїРµСЂРµРґР°С‡Р° С‚РµРєСѓС‰РµРіРѕ Р·РЅР°С‡РµРЅРёСЏ РїСЂРѕРїСЃРѕРІ*/}
      </div>
    );
  }
}
ReactDOM.render(<Clock />, document.getElementById("root"));
```

## РџРѕРґСЉРµРј СЃРѕСЃС‚РѕСЏРЅРёСЏ

```tsx
class Calculator extends React.Component {
  constructor(props) {
    super(props);
    this.handleCelsiusChange = this.handleCelsiusChange.bind(this); //РѕР±СЂР°Р±РѕС‚С‡РёРє РґР»СЏ РїРѕР»СЏ Р¦РµР»СЊСЃРёСЏ, РєРѕС‚РѕСЂС‹Р№ РјРµРЅСЏРµС‚ СЃРѕСЃС‚РѕСЏРЅРёРµ
    this.handleFahrenheitChange = this.handleFahrenheitChange.bind(this); //РѕР±СЂР°Р±РѕС‚С‡РёРє РґР»СЏ РїРѕР»СЏ Р¤Р°СЂРµРЅРіРµР№С‚
    this.state = { temperature: "", scale: "c" };
  } //РІ СЃРѕСЃС‚РѕСЏРЅРёРё РїРѕР»Рµ С‚РµРєСѓС‰РµР№ С‚РµРјРїРµСЂР°С‚СѓСЂС‹ Рё С€РєР°Р»С‹ РґР»СЏ СѓРєР°Р·Р°РЅРёСЏ С‚РѕРіРѕ, С‡С‚Рѕ РјРµРЅСЏРµС‚СЃСЏ}

  handleCelsiusChange(temperature) {
    this.setState({ scale: "c", temperature });
  } //РµСЃР»Рё РІРІРѕРґ РїСЂРѕРёСЃС…РѕРґРёС‚ Р·РґРµСЃСЊ С‚Рѕ СЃРѕСЃС‚РѕСЏРЅРёРµ РјРµРЅСЏРµС‚СЃСЏ РЅР° scale: вЂcвЂ™ Рё СѓСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ С‚РµРєСѓС‰РµРµ Р·РЅР°С‡РµРЅРёРµ С‚РµРјРїРµСЂР°С‚СѓСЂС‹ РІ Р¦РµР»СЊСЃРёСЏ
  handleFahrenheitChange(temperature) {
    this.setState({ scale: "f", temperature });
  }
  //РµСЃР»Рё РІРІРѕРґ РїСЂРѕРёСЃС…РѕРґРёС‚ Р·РґРµСЃСЊ С‚Рѕ СЃРѕСЃС‚РѕСЏРЅРёРµ РјРµРЅСЏРµС‚СЃСЏ РЅР° scale: вЂfвЂ™ Рё СѓСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ Р·РЅР°С‡РµРЅРёРµ С‚РµРјРїРµСЂР°С‚СѓСЂС‹ РІ С„Р°СЂРµРЅРіРµР№С‚Р°С…
  render() {
    const scale = this.state.scale; //РїСЂРёСЃРІР°РёРІР°РµРј РїРµСЂРµРјРµРЅРЅРѕР№ scale Р·РЅР°С‡РµРЅРёРµ РёР· СЃРѕСЃС‚РѕСЏРЅРёСЏ
    const temperature = this.state.temperature; //РїСЂРёСЃРІР°РёРІР°РµРј РїРµСЂРµРјРµРЅРЅРѕР№ temperature Р·РЅР°С‡РµРЅРёРµ РёР· СЃРѕСЃС‚РѕСЏРЅРёСЏ
    const celsius =
      scale === "f" ? tryConvert(temperature, toCelsius) : temperature; //СЂРµРЅРґРµСЂРёС‚СЃСЏ РґРІР° РєРѕРјРїРѕРЅРµРЅС‚Р° СЃРѕ Р·РЅР°С‡РµРЅРёРµРј scale === f Рё СЃРѕ Р·РЅР°С‡РµРЅРёРµРј scale === c, РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ СЌС‚РѕРіРѕ РїСЂРѕРёСЃС…РѕРґРёС‚ РєРѕРЅРІРµСЂС‚Р°С†РёСЏ
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

// <TemperatureInput scale="cвЂњ temperature={celsius} onTemperatureChange={this.handleCelsiusChange}/>
// <TemperatureInput scale="fвЂњ temperature={fahrenheit} onTemperatureChange={this.handleFahrenheitChange}/>
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
        <legend>Р’РІРµРґРёС‚Рµ РіСЂР°РґСѓСЃС‹ РїРѕ С€РєР°Р»Рµ {scaleNames[scale]}:</legend>
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
  //С„СѓРЅРєС†РёСЏ РґР»СЏ РѕР±СЂР°Р±РѕС‚РєРё СЃР»СѓС‡Р°СЏ РІРІРѕРґР° РЅРµ С‡РёСЃР»РµРЅРЅС‹С… Р·РЅР°С‡РµРЅРёР№ Рё РѕРєСЂСѓРіР»РµРЅРёСЏ СЂРµР·СѓР»СЊС‚Р°С‚Р°
  const input = parseFloat(temperature);
  if (Number.isNaN(input)) {
    return "";
  }
  const output = convert(input); // Р·Р°РґРµР№СЃС‚РІСѓРµРј РєРѕР»Р»Р±РµРє
  const rounded = Math.round(output * 1000) / 1000;
  return rounded.toString();
}
function BoilingVerdict(props) {
  //РєРѕРјРїРѕРЅРµРЅС‚ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ Р·Р°РєРёРїР°РЅРёСЏ
  if (props.celsius >= 100) {
    return <p>Р’РѕРґР° Р·Р°РєРёРїРёС‚.</p>;
  }
  return <p>Р’РѕРґР° РЅРµ Р·Р°РєРёРїРёС‚.</p>;
}
```

# Styling

## Inline СЃС‚РёР»Рё

```tsx
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
          <ToDoList value={this.state.display} />В 
        </div>В 
      </div>
    );
  }
}
```

## РћР±СЉРµРєС‚ СЃС‚РёР»РµР№

```tsx
import React, { Component } from "react";
import "./App.css";
class App extends Component {
  render() {
    const divStyle = {
      textAlign: "center", //РјРѕР¶РЅРѕ Рё вЂtext alignвЂ™ : вЂcenterвЂ™ РЅРѕ РІ РєРѕРЅСЃРѕР»Рё РІС‹Р»РµР·СѓС‚ РѕС€РёР±РєРё РїРѕ РёСЃРїСЂР°РІР»РµРЅРёСЋ РЅР° camelCase
    };
    return (
      <div _style={divStyle}>
        <h1 _style={{ color: "blue", fontSize: "20px" }}>Hello</h1>В 
      </div>
    );
  }
}
export default App;
```

## РћР±СЉРµРєС‚ СЃС‚РёР»РµР№ РІ СЂР°Р·РґРµР»СЊРЅС‹С… С„Р°Р№Р»Р°С…

```tsx
//styles.ts
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
  //РѕР±СЉРµРґРёРЅРёРј РІ РѕРґРёРЅ РѕР±СЉРµРєС‚ РґР»СЏ СЌРєСЃРїРѕСЂС‚Р°, РІ РїСЂРѕС‚РёРІРЅРѕРј СЃР»СѓС‡Р°Рµ РЅСѓР¶РЅРѕ Р±СѓРґРµС‚ РІСЃРµ СЌРєСЃРїРѕСЂС‚РёСЂРѕРІР°С‚СЊ РѕС‚РґРµР»СЊРЅРѕ
  TodoComponent: TodoComponent,
  Header: Header,
  ErrorMessage: ErrorMessage
}

export styles
```

```tsx
// app.js;
// Import the styles
import { styles } from "./styles";
class ToDoApp extends React.Component {
  // ...
  render() {
    return (
      <div _style={styles.TodoComponent}>
        <h2 _style={styles.Header}>ToDo</h2>В 
        <div>
          <Input onChange={this.handleChange} />В  В 
          <p _style={styles.ErrorMessage}>{this.state.error}</p>
          <ToDoList value={this.state.display} />В 
        </div>
      </div>
    );
  }
}
```
