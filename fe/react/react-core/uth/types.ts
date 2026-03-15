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

export type FiberNode = {};

export type FiberRootNode = {
  callbackNode: null;
  callbackProperty: number;
  // TODO
};
