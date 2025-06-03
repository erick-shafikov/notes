//функционал thunk взятый с github

const createThunkMiddleware =
  (extraArgument) =>
  ({ dispatch, getState }) =>
  (next) =>
  (action) => {
    if (typeof action === "function") {
      return action(dispatch, getState, extraArgument);
    }

    return next(action);
  };

export const thunk = createThunkMiddleware();
export const withExtraArgument = createThunkMiddleware;

// middleware => ({dispatch, getState}) => next => action =
// typeof action === 'function' => action(dispatch, getState, extraArgument)
// typeof action !== 'function' => return next(action)
