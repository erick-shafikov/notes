# props

```tsx
interface WelcomeProps {
  name: string;
}

function Welcome({ name }: WelcomeProps): JSX.Element {
  return <h1>РџСЂРёРІРµС‚, {props.name}</h1>;
}
const element = <Welcome name="РђР»РёСЃР°" />;
ReactDOM.render(element, document.getElementById("root"));
```

!!!РџСЂРёРјРµС‡Р°РЅРёРµ: Р’СЃРµРіРґР° РЅР°Р·С‹РІР°Р№С‚Рµ РєРѕРјРїРѕРЅРµРЅС‚С‹ СЃ Р·Р°РіР»Р°РІРЅРѕР№ Р±СѓРєРІС‹. Р•СЃР»Рё РєРѕРјРїРѕРЅРµРЅС‚ РЅР°С‡РёРЅР°РµС‚СЃСЏ СЃ РјР°Р»РµРЅСЊРєРѕР№ Р±СѓРєРІС‹, React РїСЂРёРЅРёРјР°РµС‚ РµРіРѕ Р·Р° DOM-С‚РµРі. РќР°РїСЂРёРјРµСЂ, div СЌС‚Рѕ div-С‚РµРі РёР· HTML, Р° Welcome СЌС‚Рѕ СѓР¶Рµ РЅР°С€ РєРѕРјРїРѕРЅРµРЅС‚ Welcome, РєРѕС‚РѕСЂС‹Р№ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РІ РѕР±Р»Р°СЃС‚Рё РІРёРґРёРјРѕСЃС‚Рё.

```tsx
//Car.tsx
import classes from "./Car.module.css";
const inputClasses = [classes.input]; //inputClasses - СЌС‚Рѕ РјР°СЃСЃРёРІ РёР· РѕРґРЅРѕРіРѕ РѕР±СЉРµРєС‚Р° СЃС‚РёР»СЏ [Car_РїСЂРµС„РёРєСЃ1, Car_РїСЂРµС„РёРєСЃ2], СЃ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёРµРј
console.log(classes.green);
if (this.props.name !== "") {
  //РµСЃР»Рё РЅРµ РїСѓСЃС‚Р°СЏ СЃС‚СЂРѕРєР°
  inputClasses.push(classes.green); //classes.green : Car_green__1BE2p
} else {
  inputClasses.push(classes.red);
}
if (this.props.name.length > 4) {
  inputClasses.push(classes.bold);
}

<input className={inputClasses.join(" ")} />; // С‚РµРїРµСЂСЊ СЃРѕРµРґРёРЅСЏРµРј РІ СЃС‚СЂРѕС‡РєСѓ
```

# children prop

РїСЂРѕРї children РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РґР»СЏ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ, РІ РєРѕС‚РѕСЂС‹С… СЃРѕРґРµСЂР¶РёРјРѕРµ РЅРµРѕРїСЂРµРґРµР»РµРЅРЅРѕ РґРѕ СЂРµРЅРґРµСЂРёРЅРіР°

```tsx
interface FancyBorderProps {
  color: string;
  children: React.ReactNode;
}

function FancyBorder({ color, children }: FancyBorderProps): JSX.Element {
  return (
    <div className={"FancyBorder FancyBorder-" + props.color}>
      {/*Р·Р°РґР°РЅРѕ РІ css*/}
      {children}В 
    </div>
  );
}
function WelcomeDialog(): JSX.Element {
  return (
    <FancyBorder color="blue">
      {/*РІСЃРµ, С‡С‚Рѕ РЅРёР¶Рµ Р±СѓРґРµС‚ СЏРІР»СЏС‚СЊСЃСЏ props.children */}
      <h1 className="Dialog-title">
        {/*Р·Р°РґР°РЅРѕ РІ css */}
        Р”РѕР±СЂРѕ РїРѕР¶Р°Р»РѕРІР°С‚СЊ
      </h1>
      <p className="Dialog-message">
        {/*Р·Р°РґР°РЅРѕ РІ css*/}
        РЎРїР°СЃРёР±Рѕ, С‡С‚Рѕ РїРѕСЃРµС‚РёР»Рё РЅР°С€ РєРѕСЃРјРёС‡РµСЃРєРёР№ РєРѕСЂР°Р±Р»СЊ! В 
      </p>
      В 
    </FancyBorder>
  );
}
```

# keys prop

Р РµРЅРґРµСЂ СЃРїРёСЃРєР° РѕС‚ 1 РґРѕ 5

```tsx
const numbers: number[] = [1, 2, 3, 4, 5];
const listItems = numbers.map(
  (number) => <li>{number}</li> //РІРѕР·РІСЂР°С‰Р°РµРј РєРѕРјРїРѕРЅРµРЅС‚ li РґР»СЏ РєР°Р¶РґРѕРіРѕ СЌР»РµРјРµРЅС‚Р°
);
ReactDOM.render(
  <ul>{listItems}</ul>, //СѓРїР°РєРѕРІС‹РІР°РµРј РІ СЃРїРёСЃРѕРє ul
  document.getElementById("root")
);
```

```tsx
interface NumberListProps {
  numbers: number[];
}

function NumberList({ numbers }: NumberListProps): JSX.Element {
  
  const listItems = numbers.map((number) => (
    <li key={number.toString()}>
      //РѕРїСЂРµРґРµР»РµРЅРёРµ РєР»СЋС‡Р° РІСЃРµРіРґР° РїСЂРѕРёСЃС…РѕРґРёС‚ РІ map-С„СѓРЅРєС†РёРё
      {number}
    </li>
  ));
  return <ul>{listItems}</ul>;
}
export default NumberList;
//index.js
const numbers = [1, 2, 3, 4, 5]; //РјР°СЃСЃРёРІ РѕС‚РїСЂР°РІРёС‚СЃСЏ РІ РїСЂРѕРїСЃС‹
ReactDOM.render(
  <NumberList numbers={numbers} />,
  document.getElementById("root")
);
```

```tsx
// РІРєР»СЋС‡РµРЅРёРµ СЂРµР·СѓР»СЊС‚Р°С‚Р° map РІРЅСѓС‚СЂРё РєРѕРјРїРѕРЅРµРЅС‚Р°

function NumberList(props) {
  
  return (
    <ul>
      {numbers.map((number) => (
        <ListItem key={number.toString()} value={number} />
      ))}
    </ul>
  );
}
```

```tsx
function App(): JSX.Element {
  return (
    <div>
      <Header /> {/*СЃС‚Р°С‚РёС‡РЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚ */}
      {notes.map((item) => (
        <Note title={item.title} content={item.content} key={item.key} /> //notes вЂ“ СЌРєСЃРїРѕСЂС‚РёСЂРѕРІР°РЅРЅС‹Р№ РјР°СЃСЃРёРІ РѕР±СЉРµРєС‚РѕРІ
      ))}
      <Footer /> {/*СЃС‚Р°С‚РёС‡РЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚ */}
    </div>
  );
}

// export default App;
//Note.jsx
interface NoteProps { title: string; content: string }

function Note({ title, content }: NoteProps): JSX.Element {
  return (
    <div className="note">
      <h1>{title}</h1>
      <p>{content}</p>
    </div>
  );
}
// export default Note;
```

## keys. BP

Р§С‚Рѕ Р±С‹ РІС‹Р·С‹РІР°С‚СЊ СЂРµРЅРґРµСЂ РєРѕРјРїРѕРЅРµРЅС‚Р° РєР°Р¶РґС‹Р№ СЂР°Р· РІ key РЅСѓР¶РЅРѕ РїРµСЂРµРґР°РІР°С‚СЊ РїСЂРѕРёР·РІРѕР»СЊРЅРѕРµ С‡РёСЃР»Рѕ
