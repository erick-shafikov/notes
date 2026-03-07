# Chrome

load , unload и visibilitychange - позволяют реагировать на изменения жизненного цикла состояния инициированного пользователем

Состояния:

- Active - страница активна и имеет фокус
- Passive - страницы видима, но без фокуса
- Hidden - если не видна
- Frozen - останавливает выполнение всех задач
- Terminated - браузер начал выгрузку страницы из памяти
- Discarded - страница видна пользователю, но удалена (видны вкладки и значок)

События:

- focus - элемент DOM получил фокус
- blur - потерял фокус
- visibilitychange - переход на новую страницу
- freeze - страница заморожена
- resume - возобновление замороженной страницы
- pageshow - переход к записи истории сеанса, загрузка новой, взятие из кеша (свойство persisted === true)
- pagehide - переход на другую страницы, положить страницу в кеш (свойство persisted === true)
- beforeunload - документ виден, нужно использовать оповещение если есть несохраненные изменения
- unload - страницы выгружается из памяти, лучше не использовать

```js
const getState = () => {
  if (document.visibilityState === "hidden") {
    return "hidden";
  }
  if (document.hasFocus()) {
    return "active";
  }
  return "passive";
};

// Stores the initial state using the `getState()` function (defined above).
let state = getState();

// Accepts a next state and, if there's been a state change, logs the
// change to the console. It also updates the `state` value defined above.
const logStateChange = (nextState) => {
  const prevState = state;
  if (nextState !== prevState) {
    console.log(`State change: ${prevState} >>> ${nextState}`);
    state = nextState;
  }
};

// Options used for all event listeners.
const opts = { capture: true };

// These lifecycle events can all use the same listener to observe state
// changes (they call the `getState()` function to determine the next state).
["pageshow", "focus", "blur", "visibilitychange", "resume"].forEach((type) => {
  window.addEventListener(type, () => logStateChange(getState()), opts);
});

// The next two listeners, on the other hand, can determine the next
// state from the event itself.
window.addEventListener(
  "freeze",
  () => {
    // In the freeze event, the next state is always frozen.
    logStateChange("frozen");
  },
  opts,
);

window.addEventListener(
  "pagehide",
  (event) => {
    // If the event's persisted property is `true` the page is about
    // to enter the back/forward cache, which is also in the frozen state.
    // If the event's persisted property is not `true` the page is
    // about to be unloaded.
    logStateChange(event.persisted ? "frozen" : "terminated");
  },
  opts,
);
```

- !!! никогда не используйте событие unload в современных браузерах.
