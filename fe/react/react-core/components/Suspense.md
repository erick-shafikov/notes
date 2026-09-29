# Suspense

- react не хранит значения между рендерами, которые были приостановлены
- fallback будет отображаться если обновление не было вызвано startTransition или useDeferredValue.
- не может определить момент когда считываются данные в эффекте и функции очистки эффекта
- если в suspense-компонент вложены несколько компонентов, то fallback будет срабатывать для каждого

```jsx
import { Suspense } from "react";
import Albums from "./Albums.js";

export default function ArtistPage({ artist }) {
  return (
    <>
      <h1>{artist.name}</h1>
      <Suspense fallback={<Loading />}>
        <Albums artistId={artist.id} />
      </Suspense>
    </>
  );
}

function Loading() {
  return <h2>🌀 Loading...</h2>;
}
```

Работает в паре с use

```tsx
export default function Speakers() {
  // функция с промисом
  async function fetchSpeakers() {
    await sleep(2000);
    const response = await fetch("http://localhost:3000/api/speakers");
    return await response.json();
  }

  const speakerListPromise = fetchSpeakers();

  return (
    <div className="container">
      <Header />
      <div className="full-page-border app-content-background">
        <Nav />
        <ErrorBoundary fallback={<div>Error Retrieving Speakers Data</div>}>
          <Suspense fallback={<div>Loading......</div>}>
            // данные на отображение
            <div className="container pb-4">
              <div className="row g-4">
                {speakerList.map(function (speaker: Speaker) {
                  return <SpeakerDetail key={speaker.id} speaker={speaker} />;
                })}
              </div>
            </div>
          </Suspense>
        </ErrorBoundary>
      </div>
      <Footer />
    </div>
  );
}
```

Мы можем декларативно определять, что React должен показывать, когда часть дерева еще не готова к рендерингу
