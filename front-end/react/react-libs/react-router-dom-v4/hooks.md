# matchPath;

# useHistory;

Обертка на history api

```js
const history = useHistory();

location = {
  action: "POP",
  block: block(prompt),
  createHref: createHref(location),
  go: go(n),
  goBack: goBack(),
  goForward: goForward(),
  length: 8,
  listen: listen(listener),
  location: {
    pathname: "/dashboard/registries",
    search: "",
    hash: "",
    state: null,
    key: "4tzy5z",
  },
  push: push(path, state),
  replace: replace(path, state),
};
```

# useLocation;

возвращает информацию о текущей локации

```js
const location = useLocation();

location = {
  hash: "";
  key: "4tzy5z";
  pathname: "/dashboard/registries";
  search: "";
  state: null;
}
```

# useParams;

```js
const params = useParams();

// если это location/entities/3 для если это location/entities/:id
params = {
  id: 3,
};
```

# useRouteMatch;
