РџР»СЋСЃС‹:

-РЈСЃРєРѕСЂРµРЅРёРµ РїРµСЂРІРѕР№ Р·Р°РіСЂСѓР·РєР° СЃС‚СЂР°РЅРёС†С‹ Р±Р»Р°РіРѕРґР°СЂСЏ СЃРµСЂРІРµСЂРЅРѕРјСѓ СЂРµРЅРґРµСЂРёРЅРіСѓ.
-РЈР»СѓС‡С€РµРЅРёРµ SEO вЂ” РїРѕРёСЃРєРѕРІРёРєРё СЃСЂР°Р·Сѓ РІРёРґСЏС‚ РєРѕРЅС‚РµРЅС‚.
-Р’РѕР·РјРѕР¶РЅРѕСЃС‚СЊ РІС‹РїРѕР»РЅСЏС‚СЊ СЂРµСЃСѓСЂСЃРѕС‘РјРєРёРµ Р·Р°РґР°С‡Рё вЂ” С‚РѕС‚ Р¶Рµ РїР°СЂСЃРёРЅРі РґР°РЅРЅС‹С… вЂ” РЅР° СЃРµСЂРІРµСЂРµ.

РњРёРЅСѓСЃС‹

- РќСѓР¶РЅР° РїРѕРґРґРµСЂР¶РєР° С„СЂРµР№РјРІРѕСЂРєР° (РЅР°РїСЂРёРјРµСЂ, Next.js), РїРѕС‚РѕРјСѓ С‡С‚Рѕ React СЃР°Рј РїРѕ СЃРµР±Рµ РЅРµ СѓРїСЂР°РІР»СЏРµС‚ СЃРµСЂРІРµСЂРѕРј.
- РќРµ РІСЃРµ РєРѕРјРїРѕРЅРµРЅС‚С‹ РјРѕР¶РЅРѕ СЃРґРµР»Р°С‚СЊ СЃРµСЂРІРµСЂРЅС‹РјРё вЂ” С‚Рµ, С‡С‚Рѕ РёСЃРїРѕР»СЊР·СѓСЋС‚ window РёР»Рё РґСЂСѓРіРёРµ Р±СЂР°СѓР·РµСЂРЅС‹Рµ API, РѕСЃС‚Р°РЅСѓС‚СЃСЏ РєР»РёРµРЅС‚СЃРєРёРјРё.
- РЈСЃР»РѕР¶РЅСЏРµС‚ Р°СЂС…РёС‚РµРєС‚СѓСЂСѓ, РµСЃР»Рё РјС‹ РЅРµ РїСЂРёРІС‹РєР»Рё Рє СЂР°Р·РґРµР»РµРЅРёСЋ СЃРµСЂРІРµСЂ-РєР»РёРµРЅС‚.

```tsx
// server-component.tsx
"use server";
export async function fetchUserData() {
  const res = await fetch("https://api.example.com/user");
  return res.json();
}
// page.js
import { fetchUserData } from "./server-component";
interface User { name: string; bio: string }

export default async function UserProfile(): Promise<JSX.Element> {
  const user = await fetchUserData();
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.bio}</p>
    </div>
  );
}
```

# RSP

РїРµСЂРµРґР°С‡Р° РјРµР¶РґСѓ СЃРµСЂРІРµСЂРѕРј Рё РєР»РёРµРЅС‚РѕРј РїСЂРѕРёСЃС…РѕРґРёС‚ СЃ РїРѕРјРѕС‰СЊ RSP - react server payload. РРјРµРµС‚ JSON РїРѕРґРѕР±РЅС‹Р№ СЃРёРЅС‚Р°РєСЃРёСЃ, РєРѕС‚РѕСЂС‹Р№ С…Р°СЂР°РєС‚РµСЂРёР·СѓРµС‚ РґРµСЂРµРІРѕ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ

```tsx
// СЃРµСЂРІРµСЂРЅС‹Р№ РєРѕРјРїРѕРЅРµРЅС‚
import { Counter } from "./client";

export default function App() {
  return (
    <div>
      <h1>Counter</h1>
      <Counter initialCount={0} />
    </div>
  );
}
```

```json
// РєРѕС‚РѕСЂС‹Р№ РІ СЃРІРѕСЋ РѕС‡РµСЂРµРґСЊ РїСЂРµРІСЂР°С‚РёС‚СЃСЏ РїСЂРё РїРµСЂРµРґР°С‡Рё
{
  "0": [
    "$",
    "div",
    null,
    {
      "children": [
        ["$", "h1", null, { "children": "Counter" }],
        ["$", "$L1", null, { "initialCount": 0 }]
      ]
    }
  ]
}
```

```tsx
// РєР»РёРµРЅС‚СЃРєРёР№ РєРѕРјРїРѕРЅРµРЅС‚
import { useState } from "react";

interface CounterProps { initialCount: number }

export function Counter({ initialCount }: CounterProps): JSX.Element {
  const [count, setCount] = useState<number>(initialCount);

  return (
    <div>
      <p>Count: {count}</p>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={() => setCount((c) => c - 1)}>в€’</button>
        <button onClick={() => setCount((c) => c + 1)}>+</button>
      </div>
    </div>
  );
}
```

Р±СѓРґРµС‚ РёРјРµС‚СЊ РІРёРґ (СЌС‚РѕС‚ payload РїСЂРёС…РѕРґРёС‚ СЃРЅР°С‡Р°Р»Р°)

```json
{
  1:I["client",[],"Counter"]
}
```

С‚Р°Рє РєР°Рє РѕРЅ РєР»РёРµРЅС‚СЃРєРё, react Р·РЅР°РµС‚ С‡С‚Рѕ РѕРЅ РµСЃС‚СЊ РІ Р±Р°РЅРґР»Рµ Рё РїРѕРґСЃС‚Р°РІРёС‚ РµРіРѕ РІ СѓРєР°Р·Р°РЅРЅРѕРµ РјРµСЃС‚Рѕ СЃ СѓРєР°Р·Р°РЅРЅС‹РјРё РїСЂРѕРїСЃР°РјРё
