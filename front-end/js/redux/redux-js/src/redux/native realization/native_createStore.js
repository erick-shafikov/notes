//внутри create store
export function createStore(rootReducer, initialState) {
  // state = 0
  // subscribers = []

  let state = rootReducer(initialState, { type: "__INIT__" }); //вызываем 1 раз для инициализации
  const subscribers = []; //вызываем 1 раз для инициализации
  return {
    //по итогу createStore возвращает объект из 3 функций
    dispatch(action) {
      state = rootReducer(state, action); //взрывается соответствующее действие и переопределяется state
      subscribers.forEach((sub) => sub());
    },
    subscribe(callback) {
      subscribers.push(callback);
    },
    getState() {
      return state;
    },
  };
}
