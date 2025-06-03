import "./styles.css";
import { applyMiddleware, createStore } from "redux";
import { composeWithDevTools } from "@redux-devtools/extension";
import { rootReducer } from "./redux/rootReducer";
import thunk from "redux-thunk";
import {
  asyncIncrement,
  changeTheme,
  decrement,
  increment,
  incrementByAmount,
} from "./redux/actions";
import { createLogger } from "redux-logger";

const counter = document.getElementById("counter");
const addBtn = document.getElementById("add");
const subBtn = document.getElementById("sub");
const asyncBtn = document.getElementById("async");
const themeBtn = document.getElementById("theme");
const addSome = document.getElementById("add_some");
const addField = document.getElementById("add_field");

/*import { compose } from 'redux';//для расширения middleware под redux-dev-tools
const store = createStore(
    rootReducer,
    compose(
        applyMiddleware(thunk, logger),
        window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__()
    )
);
*/

/*--------------------------------------(1) создание store--------------------------------------------*/
const store = createStore(
  /*создаем store, принимаем аргументы (rootReducer, в данном случаем комбинация reducer и state) функционал:*/
  rootReducer, //принимаем тело функций reducer в которых есть функционал действий И ВОЗВРАЩАЕТ НОВОЕ СОСТОЯНИЕ
  /*
    //результат вызова createStore
    store = {
        #state: 0, //явно, приватное поле
        #subscribers: [],//явно, приватное поле, еще не инициализировали
        #rootReducer = {//неявно, так как roodReducer берется из аргументов createStore
            counter: function(state = 0, action){
                ..//counter reducer
            },
            theme : function {
                ...//theme reducer
            }
        }
    }
    */
  composeWithDevTools(
    applyMiddleware(thunk) //включаем промежуточные обработчики
  )
);
/*возвращает в переменную store {
    #state: 0, //в нашем случае это просто число
    #subscribers: [],
    dispatch(action){
        state = rootReducer(state, action);//вызывается соответствующее действие
        subscribers.forEach(sub => sub());
    },
    subscribe(callback) {
        subscribers.push(callback);
    },
    getState() {
        return state
    }

} */

addBtn.addEventListener("click", () => {
  store.dispatch(increment());
});
/*(4) Вызов 
при нажатии на кнопку вызывается диспетчер с аргументом действия {type: INCREMENT}
    - из объекта store достаем dispatch c аргументом { type: INCREMENT }
    - он достает оттуда состояние rootReducer в который кладет старый state и с учетом функционала возвращает новое состояние
    - state в store.dispatch принимает новое значение
    - каждый из слушателей получает новое значение state (2)
*/
/*переменной store {
    #state: 0,
    #subscribers: [
        () => {const state = store.getState();...},
    ],
    dispatch(increment() : 'INCREMENT'){//вызываем из dispatch action == 'INCREMENT',
        state = rootReducer(state : 0, action: 'INCREMENT');
        /*вызывается соответствующее действие и ПЕРЕПИСЫВАЕТСЯ ПРИВАТНОЕ ЗНАЧЕНИЕ STATE
        subscribers.forEach(sub => sub());//в каждый подписчик отправляется это значение
    },
    subscribe(callback) {
        subscribers.push(callback);
    },
    getState() {
        return state
    }

} */
subBtn.addEventListener("click", () => {
  store.dispatch(decrement());
});

asyncBtn.addEventListener("click", () => {
  store.dispatch(asycIncreament());
});

themeBtn.addEventListener("click", () => {
  const newTheme = document.body.classList.contains("light") ? "dark" : "light";
  store.dispatch(changeTheme(newTheme));
});

addSome.addEventListener("click", () => {
  store.dispatch(incrementByAmount(addfield.value));
});

/*--------------------Запускается один раз при начальном пуске приложения--------------------------------------- */
//(2)
store.subscribe(() => {
  const state = store.getState();
  counter.textContent = state.counter;
  document.body.className = state.theme.value;

  [addBtn, subBtn, themeBtn, asyncBtn].forEach((btn) => {
    btn.disabled = state.theme.disabled;
  });
});

/*вызов функции слушателя, который вызывает перерендер, НО на первой инициализации, добавляется в массив слушателей
callback который достает текущее состояние: 
- отрисовывает counter, определяет тему
- добавляет тему
- определяет состояние каждой кнопки 
*/

/*возвращает в переменную store {
    #state: 0, //в нашем случае это просто число
    #subscribers: [
        () => {//кладем в #subscribers функционал обновления
            const state = store.getState();
            counter.textContent = state.counter;
            document.body.className = state.theme.value;

            [addBtn, subBtn, themeBtn, asyncBtn].forEach(btn => {
                btn.disabled = state.theme.disabled
        },
    ],
    dispatch(action){
        state = rootReducer(state, action);//взрывается соответствующее действие переопределяется state
        subscribers.forEach(sub => sub());
    },
    subscribe(callback) {
        subscribers.push(callback);
    },
    getState() {
        return state
    }

} */
//(3) инициализация состояния
store.dispatch({
  type: "init_app",
});
/*инициализация состояния приложения, активируя callback из (2):
- достает начальное стояние счетчика = 0
- достает начальное состояние темы

без него не было бы начального состояния у счетчика
*/

// store.subscribe(()=>alert(store.getState().counter))//еще одна подписка на store
