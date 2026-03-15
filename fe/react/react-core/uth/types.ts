export type FiberRoot = {};

export type ReactElement = {
  $$typeof: Symbol;
  key: string | null;
  props: {};
  ref: null;
  type: "div";
  _owner: null;
  _store: { validated: boolean };
  _self: null | ReactElement;
  _source: null;
};

export type ReactDomLegacyRoot = {
  _internalRoot: FiberNode;
};

export type FiberRootNode = {
  callbackNode: null;
  callbackProperty: number;
  // TODO
};

type FiberNode = {
  type: "div"; //Тип узла
  child: FiberNode; //Ссылка на первый дочерний узел, если он есть.
  sibling: FiberNode; //Ссылка на следующий узел этого же уровня.
  return: FiberNode; //Ссылка на родительский узел
  effectTag: "update"; //Тип обновления.
  pendingProps: {}; //Пропсы, которые должны быть применены к этому узлу
  memoizedProps: {
    cache: {}; //?закешированные значения
    element: "";
  }; //Пропсы, которые использовались для рендеринга этого компонента в прошлый раз.
  alternate: FiberNode; //Ссылка на Fiber-узел
  updateQueue: null; //
  hooks: null; //Информация о хуках
  index: 0; //Индекс элемента в родительском списке
  nextEffect: FiberNode; //Ссылка на следующий эффект в списке эффектов.
  firstEffect: FiberNode; //Ссылка на первый эффект в списке.
  lastEffect: FiberNode; //Ссылка на последний эффект в списке.
  tag: "FunctionComponent"; //Тип компонента.
  mode: 0; //Режим рендеринга  Concurrent Mode или Strict Mode
  stateNode: {
    //Ссылка на реальный DOM-узел
    hydrate: boolean;
    mutableSourceEagerHydrationData: any[];
  };
};

export type DOMNode = {
  hydrate: boolean;
  mutableSourceEagerHydrationData: any[];
};
