# render prop

- РїРµСЂРµРґР°С‡Р° РІ РєР°С‡РµСЃС‚РІРµ props С„СѓРЅРєС†РёРё, РєРѕС‚РѕСЂР°СЏ Р±СѓРґРµС‚ РїСЂРёРЅРёРјР°С‚СЊ РєР°РєРёРµ-С‚Рѕ РґР°РЅРЅС‹Рµ РѕС‚ РґРѕС‡РµСЂРЅРµРіРѕ РєРѕРјРїРѕРЅРµРЅС‚Р° Рё РѕС‚СЂРёСЃРѕРІС‹РІР°С‚СЊ РёС… С‚Р°Рє, РєР°Рє Р±СѓРґРµС‚ СѓРєР°Р·Р°РЅРѕ РІ СЂРѕРґРёС‚РµР»СЊСЃРєРѕРј

```tsx
const ParentComponent = (): JSX.Element => {
  //РїРѕР»СѓС‡Р°РµРј РІ РєР°С‡РµСЃС‚РІРµ РїСЂРѕРїСЃР° С„СѓРЅРєС†РёСЋ РЅР° РѕС‚СЂРёСЃРѕРІРєСѓ СЃ Р°СЂРіСѓРјРµРЅС‚Р°РјРё
  return <ChildComponent render={(text) => <h1>{text}</h1>} />;
};

interface ChildComponentProps { render: (text: string) => React.ReactNode }

const ChildComponent = ({ render }: ChildComponentProps): JSX.Element => {
  const text = "Hello World";
  //Р°СЂРіСѓРјРµРЅС‚Р°РјРё, РєРѕС‚РѕСЂС‹Рµ РјРѕР¶РЅРѕ РїРµСЂРµРґР°С‚СЊ РёР· РґРѕС‡РµСЂРЅРµРіРѕ
  return <div>{render(text)}</div>;
};
```

counter

```tsx
interface ClickCounterProps { render: (state: { count: number; increment: () => void }) => React.ReactNode }

const ClickCounter = ({ render }: ClickCounterProps): JSX.Element => {
  const [count, setCount] = useState<number>(0);

  const increment = () => setCount(count + 1);

  return <div>{render({ count, increment })}</div>;
};

<ClickCounter
  render={({ count, increment }) => (
    <div>
      <h2>РљР°СЃС‚РѕРјРЅС‹Р№ СЃС‡С‘С‚С‡РёРє</h2>
      <p>РљРѕР»РёС‡РµСЃС‚РІРѕ РєР»РёРєРѕРІ: {count}</p>
      <button onClick={increment}>РџСЂРёР±Р°РІРёС‚СЊ 1</button>
    </div>
  )}
/>;
```

Р¤РѕСЂРјР°, РІ РєРѕС‚РѕСЂРѕР№ РµСЃР»Рё РЅРµ РїРµСЂРµРґР°РЅ render РѕС‚СЂРёСЃСѓРµС‚ С„РѕСЂРјСѓ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ

```tsx
interface FormValues { [key: string]: string }
interface FormProps { initialValues: FormValues; render?: (props: { values: FormValues; handleChange: React.ChangeEventHandler<HTMLInputElement>; handleSubmit: React.FormEventHandler<HTMLFormElement> }) => React.ReactNode }

const Form = ({ initialValues, render }: FormProps): JSX.Element => {
  const [values, setValues] = useState<FormValues>(initialValues);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const { name, value } = event.target;
    setValues((previousValues) => ({ ...previousValues, [name]: value }));
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    console.log("РћС‚РїСЂР°РІР»РµРЅРЅС‹Рµ Р·РЅР°С‡РµРЅРёСЏ", values);
  };

  //РµСЃР»Рё РїСЂРµРґР°РЅ, С‚Рѕ РѕС‚СЂРёСЃСѓРµС‚СЃСЏ С„РѕСЂРјР°, РєРѕС‚РѕСЂСѓСЋ РїРµСЂРµРґР°Р»Рё
  if (render) {
    return render({
      values,
      handleChange,
      handleSubmit,
    });
  }

  //РёРЅР°С‡Рµ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ
  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(initialValues).map((key) => (
        <div key={key}>
          <label>
            <div>{key[0].toUpperCase() + key.slice(1)}:</div>
            <input
              type="text"
              name={key}
              value={values[key]}
              onChange={handleChange}
            />
          </label>
        </div>
      ))}
      <button type="submit">РћС‚РїСЂР°РІРёС‚СЊ</button>
    </form>
  );
};
```
