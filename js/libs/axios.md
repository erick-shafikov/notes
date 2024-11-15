# axios

Запрос можно осуществить с помощью статических методов

```js
axios
  .post("/user", {
    firstName: "Fred",
    lastName: "Flintstone",
  })
  .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
```

или с помощью создания экземпляра

```js
const instance = axios.create({
  baseURL: "https://some-domain.com/api/",
  timeout: 1000,
  headers: { "X-Custom-Header": "foobar" },
});
```

**Конфигурация запроса**

```js
const config = {
  url: "/user",
  method: "get",
  baseURL: "https://some-domain.com/api",
  //функция по трансформации запроса
  transformRequest: [
    //data - тело запроса, headers - заголовки
    function (data, headers) {
      return data;
    },
  ],
  //изменить ответ
  transformResponse: [
    function (data) {
      return data;
    },
  ],
  headers: { "X-Requested-With": "XMLHttpRequest" }, //установка заголовков
  // параметры запроса
  params: {
    ID: 12345,
  },
  // функция которая сериализует параметры
  paramsSerializer: function (params) {
    return Qs.stringify(params, { arrayFormat: "brackets" });
  },

  // `data` is the data to be sent as the request body
  // Only applicable for request methods 'PUT', 'POST', 'DELETE', and 'PATCH'
  // When no `transformRequest` is set, must be of one of the following types:
  // - string, plain object, ArrayBuffer, ArrayBufferView, URLSearchParams
  // - Browser only: FormData, File, Blob
  // - Node only: Stream, Buffer
  data: {
    firstName: "Fred",
  },
  data: "Country=Brasil&City=Belo Horizonte",
  // для сброса запроса
  timeout: 1000, // default is `0` (no timeout)
  // `withCredentials` indicates whether or not cross-site Access-Control requests
  withCredentials: false, // default
  // `adapter` allows custom handling of requests which makes testing easier.
  // Return a promise and supply a valid response (see lib/adapters/README.md).
  adapter: function (config) {
    /* ... */
  },

  // `auth` indicates that HTTP Basic auth should be used, and supplies credentials.
  // This will set an `Authorization` header, overwriting any existing
  // `Authorization` custom headers you have set using `headers`.
  // Please note that only HTTP Basic auth is configurable through this parameter.
  // For Bearer tokens and such, use `Authorization` custom headers instead.
  auth: {
    username: "janedoe",
    password: "s00pers3cret",
  },
  // `responseType` indicates the type of data that the server will respond with
  // options are: 'arraybuffer', 'document', 'json', 'text', 'stream'
  //   browser only: 'blob'
  responseType: "json", // default
  responseEncoding: "utf8", // default
  // `xsrfCookieName` is the name of the cookie to use as a value for xsrf token
  xsrfCookieName: "XSRF-TOKEN", // default
  // `xsrfHeaderName` is the name of the http header that carries the xsrf token value
  xsrfHeaderName: "X-XSRF-TOKEN", // default
  // `onUploadProgress` allows handling of progress events for uploads
  // browser only
  onUploadProgress: function (progressEvent) {
    // Do whatever you want with the native progress event
  },
  // `onDownloadProgress` allows handling of progress events for downloads
  // browser only
  onDownloadProgress: function (progressEvent) {
    // Do whatever you want with the native progress event
  },
  maxContentLength: 2000,
  maxBodyLength: 2000,
  // валидирует статус
  validateStatus: function (status) {
    return status >= 200 && status < 300; // default
  },
  maxRedirects: 5, // default
  socketPath: null, // default
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
  proxy: {
    protocol: "https",
    host: "127.0.0.1",
    port: 9000,
    auth: {
      username: "mikeymike",
      password: "rapunz3l",
    },
  },
  cancelToken: new CancelToken(function (cancel) {}),
  decompress: true, // default
};
```

**Конфигурация ответа**

```js
const response = {
  data: {},
  status: 200,
  statusText: "OK",
  headers: {},

  // config переданный в axios в запросе
  config: {},
  // `request` is the request that generated this response
  // It is the last ClientRequest instance in node.js (in redirects)
  // and an XMLHttpRequest instance in the browser
  request: {},
};
```

Переопределение значений

```js
const instance = axios.create({
  baseURL: "https://api.example.com",
});

instance.defaults.headers.common["Authorization"] = AUTH_TOKEN;

instance.defaults.timeout = 2500;
// переопределение для определенного типа запроса
instance.get("/longRequest", {
  timeout: 5000,
});
```

**Интерцепторы**

```js
// Добавляем интерцептор на запрос
axios.interceptors.request.use(
  function (config) {
    // Обработчик запросов
    return config;
  },
  function (error) {
    // обработчик ошибок
    return Promise.reject(error);
  }
);
// интерцептор на ответ
axios.interceptors.response.use(
  function (response) {
    //обработка данных ответа
    return response;
  },
  function (error) {
    // обработчик ошибок
    return Promise.reject(error);
  }
);
//добавляем
const instance = axios.create();
instance.interceptors.request.use(function () {
  /*...*/
});
```

Удаление

```js
const myInterceptor = axios.interceptors.request.use(function () {
  /*...*/
});
axios.interceptors.request.eject(myInterceptor);
```

**Обработка ошибок**

```js
axios.get("/user/12345").catch(function (error) {
  if (error.response) {
    //поля ошибки
    console.log(error.response.data);
    console.log(error.response.status);
    console.log(error.response.headers);
  } else if (error.request) {
    //поля запроса ошибки
    console.log(error.request);
  } else {
    console.log("Error", error.message);
  }
  //подробное описание ошибки
  console.log(error.config);
});
```

**отмена запроса**

Отменить можно с помощью AbortController

```js
const controller = new AbortController();

axios
  .get("/foo/bar", {
    signal: controller.signal,
  })
  .then(function (response) {
    //...
  });
// cancel the request
controller.abort();
```
