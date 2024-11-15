<!--ulbi------------------------------------------------------------------------------------------------------------------------------------------>

# Ulbi

## Ulbi. Настройка Webpack

Запуск проекта осуществляется с помощью команды webpack

```json
{
  "build:prod": "webpack  --env mode=production",
  "build:dev": "webpack  --env mode=development"
}
```

```ts
//webpack.config.ts в корне
// для типизации
import webpack from "webpack";
// утилита для определения путей
import path from "path";
// Сам конфиг
import { buildWebpackConfig } from "./config/build/buildWebpackConfig";
// Типизация
import { BuildEnv, BuildMode, BuildPaths } from "./config/build/types/config";

// утилита для url
function getApiUrl(mode: BuildMode, apiUrl?: string) {
  if (apiUrl) {
    return apiUrl;
  }
  if (mode === "production") {
    return "/api";
  }

  return "http://localhost:8000";
}

export default (env: BuildEnv) => {
  // пути до статики
  const paths: BuildPaths = {
    entry: path.resolve(__dirname, "src", "index.tsx"),
    build: path.resolve(__dirname, "build"),
    html: path.resolve(__dirname, "public", "index.html"),
    src: path.resolve(__dirname, "src"),
    locales: path.resolve(__dirname, "public", "locales"),
    buildLocales: path.resolve(__dirname, "build", "locales"),
  };
  // переменные окружения
  const mode = env?.mode || "development"; //"build:prod": "webpack  --env mode=production
  const PORT = env?.port || 3000;
  const apiUrl = getApiUrl(mode, env?.apiUrl);

  const isDev = mode === "development";
  // функция сборщик конфига
  const config: webpack.Configuration = buildWebpackConfig({
    mode,
    paths,
    isDev,
    port: PORT,
    apiUrl,
    project: "frontend",
  });

  return config;
};
```

```ts
//./config/build/buildWebpackConfig
import webpack from "webpack";
import { BuildOptions } from "./types/config";
import { buildPlugins } from "./buildPlugins";
import { buildLoaders } from "./buildLoaders";
import { buildResolvers } from "./buildResolvers";
import { buildDevServer } from "./buildDevServer";

export function buildWebpackConfig(
  options: BuildOptions
): webpack.Configuration {
  const { paths, mode, isDev } = options;

  return {
    mode,
    entry: paths.entry, //"src", "index.tsx"
    output: {
      filename: "[name].[contenthash].js", //имя каждого файла
      path: paths.build, //куда собирать проект
      clean: true, //будет удалять предыдущие фалы компиляции
      publicPath: "/",
    },
    // функция для добавления плагинов
    plugins: buildPlugins(options),
    module: {
      rules: buildLoaders(options),
    },
    resolve: buildResolvers(options),
    devtool: isDev ? "eval-cheap-module-source-map" : undefined,
    devServer: isDev ? buildDevServer(options) : undefined, //dev сервер
  };
}
```

Типизация для утилит

```ts
export type BuildMode = "production" | "development";

export interface BuildPaths {
  entry: string;
  build: string;
  html: string;
  src: string;
  locales: string;
  buildLocales: string;
}

export interface BuildEnv {
  mode: BuildMode;
  port: number;
  apiUrl: string;
}

export interface BuildOptions {
  mode: BuildMode;
  paths: BuildPaths;
  isDev: boolean;
  port: number;
  apiUrl: string;
  project: "storybook" | "frontend" | "jest";
}
```

### Настройка Webpack. Плагины

```ts
// config/build/buildPlugins.ts
//
import HtmlWebpackPlugin from "html-webpack-plugin";
import webpack from "webpack";
import MiniCssExtractPlugin from "mini-css-extract-plugin";
import { BundleAnalyzerPlugin } from "webpack-bundle-analyzer";
import ReactRefreshWebpackPlugin from "@pmmmwh/react-refresh-webpack-plugin";
import CopyPlugin from "copy-webpack-plugin";
import CircularDependencyPlugin from "circular-dependency-plugin";
import ForkTsCheckerWebpackPlugin from "fork-ts-checker-webpack-plugin";
import { BuildOptions } from "./types/config";

export function buildPlugins({
  paths,
  isDev,
  apiUrl,
  project,
}: BuildOptions): webpack.WebpackPluginInstance[] {
  const isProd = !isDev;

  const plugins = [
    // плагин для создает html файл
    new HtmlWebpackPlugin({
      // где лежит html
      template: paths.html,
    }),
    new webpack.ProgressPlugin(),
    // Создает глобальные переменные ниже __IS_DEV__, __API__, __PROJECT__
    new webpack.DefinePlugin({
      __IS_DEV__: JSON.stringify(isDev),
      __API__: JSON.stringify(apiUrl),
      __PROJECT__: JSON.stringify(project), // "storybook" | "frontend" | "jest"
    }),
    new CircularDependencyPlugin({
      exclude: /node_modules/,
      failOnError: true,
    }),
    // для улучшения ts разработки
    new ForkTsCheckerWebpackPlugin({
      typescript: {
        diagnosticOptions: {
          semantic: true,
          syntactic: true,
        },
        mode: "write-references",
      },
    }),
  ];

  if (isDev) {
    // hot reload для react компонентов
    plugins.push(new ReactRefreshWebpackPlugin());
    plugins.push(new webpack.HotModuleReplacementPlugin());
    plugins.push(
      new BundleAnalyzerPlugin({
        openAnalyzer: false,
      })
    );
  }

  // минификация для prod окружения

  if (isProd) {
    plugins.push(
      new MiniCssExtractPlugin({
        filename: "css/[name].[contenthash:8].css",
        chunkFilename: "css/[name].[contenthash:8].css",
      })
    );
    // копирует файлы в финальный bundle
    //здесь нужен для копирования файлов с переводами
    plugins.push(
      new CopyPlugin({
        patterns: [{ from: paths.locales, to: paths.buildLocales }],
      })
    );
  }

  return plugins;
}
```

### Настройка Webpack. Модули

builder для модулей, для разделения файлов в bundle

```ts
import webpack from "webpack";
import { buildCssLoader } from "./loaders/buildCssLoader";
import { BuildOptions } from "./types/config";
import { buildBabelLoader } from "./loaders/buildBabelLoader";

export function buildLoaders(options: BuildOptions): webpack.RuleSetRule[] {
  const { isDev } = options;

  //для svg
  const svgLoader = {
    test: /\.svg$/,
    use: [
      {
        loader: "@svgr/webpack",
        options: {
          icon: true,
          svgoConfig: {
            plugins: [
              {
                name: "convertColors",
                params: {
                  currentColor: true,
                },
              },
            ],
          },
        },
      },
    ],
  };

  // опции для js
  const codeBabelLoader = buildBabelLoader({ ...options, isTsx: false });
  // опции для ts
  const tsxCodeBabelLoader = buildBabelLoader({ ...options, isTsx: true });

  const cssLoader = buildCssLoader(isDev);

  // Если не используем ts - нужен babel-loader
  // const typescriptLoader = {
  //     test: /\.tsx?$/,
  //     use: 'ts-loader',
  //     exclude: /node_modules/,
  // };

  // опции для всего остального (картинки и шрифты)
  const fileLoader = {
    test: /\.(png|jpe?g|gif|woff2|woff)$/i,
    use: [
      {
        loader: "file-loader",
      },
    ],
  };

  return [
    fileLoader,
    svgLoader,
    codeBabelLoader,
    tsxCodeBabelLoader,
    // typescriptLoader,
    cssLoader,
  ];
}
```

```ts
//config/build/loaders/buildBabelLoader.ts
import { BuildOptions } from "../types/config";
import babelRemovePropsPlugin from "../../babel/babelRemovePropsPlugin";

interface BuildBabelLoaderProps extends BuildOptions {
  isTsx?: boolean;
}

// isTsx флаг который говорит о том js это или ts
// Подключает babel loaders
export function buildBabelLoader({ isDev, isTsx }: BuildBabelLoaderProps) {
  const isProd = !isDev;
  return {
    test: isTsx ? /\.(jsx|tsx)$/ : /\.(js|ts)$/,
    exclude: /node_modules/,
    use: {
      loader: "babel-loader",
      options: {
        cacheDirectory: true,
        presets: ["@babel/preset-env"],
        plugins: [
          [
            "@babel/plugin-transform-typescript",
            {
              isTsx,
            },
          ],
          "@babel/plugin-transform-runtime",
          isTsx &&
            isProd && [
              babelRemovePropsPlugin,
              {
                props: ["data-testid"],
              },
            ],
          isDev && require.resolve("react-refresh/babel"),
        ].filter(Boolean),
      },
    },
  };
}
```

```ts
import { PluginItem } from "@babel/core";

// eslint-disable-next-line func-names
export default function (): PluginItem {
  return {
    visitor: {
      Program(path, state) {
        const forbidden = state.opts.props || [];

        path.traverse({
          JSXIdentifier(current) {
            const nodeName = current.node.name;

            if (forbidden.includes(nodeName)) {
              current.parentPath.remove();
            }
          },
        });
      },
    },
  };
}
```

```ts
// лоудер для css
import MiniCssExtractPlugin from "mini-css-extract-plugin";

export function buildCssLoader(isDev: boolean) {
  return {
    test: /\.s[ac]ss$/i,
    exclude: /node_modules/,
    use: [
      isDev ? "style-loader" : MiniCssExtractPlugin.loader,
      {
        loader: "css-loader",
        options: {
          modules: {
            auto: (resPath: string) => Boolean(resPath.includes(".module.")),
            localIdentName: isDev
              ? "[path][name]__[local]--[hash:base64:5]"
              : "[hash:base64:8]",
          },
        },
      },
      "sass-loader",
    ],
  };
}
```

```ts
// фал для добавления алиасов
import { ResolveOptions } from "webpack";
import { BuildOptions } from "./types/config";

export function buildResolvers(options: BuildOptions): ResolveOptions {
  return {
    extensions: [".tsx", ".ts", ".js"],
    preferAbsolute: true,
    modules: [options.paths.src, "node_modules"],
    mainFiles: ["index"],
    alias: {
      "@": options.paths.src,
    },
  };
}
```

```ts
//config/build/buildDevServer.ts
//функция, которая возвращает объект настройки dev server
import type { Configuration as DevServerConfiguration } from "webpack-dev-server";
import { BuildOptions } from "./types/config";

export function buildDevServer(options: BuildOptions): DevServerConfiguration {
  return {
    port: options.port,
    open: true,
    historyApiFallback: true,
    hot: true,
  };
}
```

### Настройка vite

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";

export default defineConfig({
  plugins: [svgr({ exportAsDefault: true }), react()],
  resolve: {
    alias: [{ find: "@", replacement: "/src" }],
  },
  define: {
    __IS_DEV__: JSON.stringify(true),
    __API__: JSON.stringify("http://localhost:8000"),
    __PROJECT__: JSON.stringify("frontend"),
  },
});
```

<!-- ULBI. REDUX -->

## ULBI. REDUX

Основа - метод code splitting https://redux.js.org/usage/code-splitting

```tsx
import { ReactNode } from "react";
import { Provider } from "react-redux";
import { ReducersMapObject } from "@reduxjs/toolkit";
import { createReduxStore } from "../config/store";
import { StateSchema } from "../config/StateSchema";

interface StoreProviderProps {
  children?: ReactNode;
  // редюссеры на постоянной основе
  initialState?: DeepPartial<StateSchema>;
  // редюссеры асинхронные
  asyncReducers?: DeepPartial<ReducersMapObject<StateSchema>>;
}

export const StoreProvider = (props: StoreProviderProps) => {
  const { children, initialState, asyncReducers } = props;

  // const navigate = useNavigate();

  const store = createReduxStore(
    initialState as StateSchema,
    asyncReducers as ReducersMapObject<StateSchema>
    // navigate,
  );

  return <Provider store={store}>{children}</Provider>;
};
```

### createReduxStore

```tsx
//src/app/providers/StoreProvider/ui/store.tsx
// расширенный configureStore
import { configureStore, ReducersMapObject } from "@reduxjs/toolkit";
import { CombinedState, Reducer } from "redux";
import { counterReducer } from "@/entities/Counter";
import { userReducer } from "@/entities/User";
import { $api } from "@/shared/api/api";
import { uiReducer } from "@/features/UI";
import { rtkApi } from "@/shared/api/rtkApi";
import { StateSchema, ThunkExtraArg } from "./StateSchema";
import { createReducerManager } from "./reducerManager";

export function createReduxStore(
  initialState?: StateSchema,
  asyncReducers?: ReducersMapObject<StateSchema>
) {
  const rootReducers: ReducersMapObject<StateSchema> = {
    ...asyncReducers,
    counter: counterReducer,
    user: userReducer,
    ui: uiReducer,
    [rtkApi.reducerPath]: rtkApi.reducer,
  };

  // редюссеры
  const reducerManager = createReducerManager(rootReducers);

  const extraArg: ThunkExtraArg = {
    api: $api,
  };

  const store = configureStore({
    //подключение асинхронных редюсеров
    reducer: reducerManager.reduce as Reducer<CombinedState<StateSchema>>,
    devTools: __IS_DEV__,
    preloadedState: initialState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        thunk: {
          extraArgument: extraArg,
        },
      }).concat(rtkApi.middleware),
  });

  // в store помещается reducerManager

  // @ts-ignore
  store.reducerManager = reducerManager;

  return store;
}

export type AppDispatch = ReturnType<typeof createReduxStore>["dispatch"];
```

```ts
// динамический редюсер
import {
  AnyAction,
  combineReducers,
  Reducer,
  ReducersMapObject,
} from "@reduxjs/toolkit";
import {
  MountedReducers,
  ReducerManager,
  StateSchema,
  StateSchemaKey,
} from "./StateSchema";

export function createReducerManager(
  // редюссер состояния
  initialReducers: ReducersMapObject<StateSchema>
): ReducerManager {
  const reducers = { ...initialReducers };

  // собираем reducer
  let combinedReducer = combineReducers(reducers);

  // имена редюсеров на удаление
  let keysToRemove: Array<StateSchemaKey> = [];
  // объект вида {__reducerName__: boolean}, который отображает информацию по редюсерам, которые есть
  const mountedReducers: MountedReducers = {};

  return {
    // получить редюсеры
    getReducerMap: () => reducers,
    // получить информацию о редюсерах
    getMountedReducers: () => mountedReducers,
    // обработка редюсеров, используется в корневом провайдере контекста
    reduce: (state: StateSchema, action: AnyAction) => {
      if (keysToRemove.length > 0) {
        state = { ...state };
        keysToRemove.forEach((key) => {
          delete state[key];
        });
        keysToRemove = [];
      }
      return combinedReducer(state, action);
    },
    // метод вызывается только в менеджере
    add: (key: StateSchemaKey, reducer: Reducer) => {
      if (!key || reducers[key]) {
        return;
      }
      reducers[key] = reducer;
      mountedReducers[key] = true;

      combinedReducer = combineReducers(reducers);
    },
    // метод вызывается только в менеджере
    remove: (key: StateSchemaKey) => {
      if (!key || !reducers[key]) {
        return;
      }
      delete reducers[key];
      keysToRemove.push(key);
      mountedReducers[key] = false;

      combinedReducer = combineReducers(reducers);
    },
  };
}
```

```ts
// утилита по созданию slice
import { bindActionCreators, createSlice } from "@reduxjs/toolkit";
import { SliceCaseReducers, CreateSliceOptions } from "@reduxjs/toolkit/dist";
import { useDispatch } from "react-redux";
import { useMemo } from "react";

export function buildSlice<
  State,
  CaseReducers extends SliceCaseReducers<State>,
  Name extends string = string
>(options: CreateSliceOptions<State, CaseReducers, Name>) {
  const slice = createSlice(options);

  const useActions = (): typeof slice.actions => {
    const dispatch = useDispatch();

    // @ts-ignore
    return useMemo(
      // @ts-ignore
      () => bindActionCreators(slice.actions, dispatch),
      [dispatch]
    );
  };

  return {
    ...slice,
    useActions,
  };
}
```

пример применения для build slice

```ts
import { PayloadAction } from "@reduxjs/toolkit";
import { CounterSchema } from "../types/counterSchema";
import { buildSlice } from "@/shared/lib/store";

const initialState: CounterSchema = {
  value: 0,
};

export const counterSlice = buildSlice({
  name: "counter",
  initialState,
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    add: (state, { payload }: PayloadAction<number>) => {
      state.value += payload;
    },
    decrement: (state) => {
      state.value -= 1;
    },
  },
});

export const {
  actions: counterActions,
  reducer: counterReducer,
  useActions: useCounterActions,
} = counterSlice;
```

```ts
// для создания селекторов
import { useSelector } from "react-redux";
import { StateSchema } from "@/app/providers/StoreProvider";

type Selector<T, Args extends any[]> = (state: StateSchema, ...args: Args) => T;
type Hook<T, Args extends any[]> = (...args: Args) => T;
type Result<T, Args extends any[]> = [Hook<T, Args>, Selector<T, Args>];

export function buildSelector<T, Args extends any[]>(
  selector: Selector<T, Args>
): Result<T, Args> {
  const useSelectorHook: Hook<T, Args> = (...args: Args) => {
    return useSelector((state: StateSchema) => selector(state, ...args));
  };

  return [useSelectorHook, selector];
}
```

использование динамического store. Каждый компонент оборачивается в DynamicModuleLoader

```tsx
import { ReactNode, useEffect } from "react";
import { useDispatch, useStore } from "react-redux";
import { Reducer } from "@reduxjs/toolkit";
import {
  ReduxStoreWithManager,
  StateSchema,
  StateSchemaKey,
} from "@/app/providers/StoreProvider";

export type ReducersList = {
  [name in StateSchemaKey]?: Reducer<NonNullable<StateSchema[name]>>;
};

interface DynamicModuleLoaderProps {
  reducers: ReducersList;
  removeAfterUnmount?: boolean;
  children: ReactNode;
}

export const DynamicModuleLoader = (props: DynamicModuleLoaderProps) => {
  const { children, reducers, removeAfterUnmount = true } = props;

  // получаем объект store в котором есть метод reducerManager
  const store = useStore() as ReduxStoreWithManager;
  const dispatch = useDispatch();

  useEffect(() => {
    // получаем список смонтированных
    const mountedReducers = store.reducerManager.getMountedReducers();

    // проходимся по списку
    Object.entries(reducers).forEach(([name, reducer]) => {
      const mounted = mountedReducers[name as StateSchemaKey];
      // Добавляем новый редюсер только если его нет
      if (!mounted) {
        store.reducerManager.add(name as StateSchemaKey, reducer);
        dispatch({ type: `@INIT ${name} reducer` });
      }
    });

    return () => {
      if (removeAfterUnmount) {
        // удаляем при депонировании компонента
        Object.entries(reducers).forEach(([name, reducer]) => {
          store.reducerManager.remove(name as StateSchemaKey);
          dispatch({ type: `@DESTROY ${name} reducer` });
        });
      }
    };
    // eslint-disable-next-line
  }, []);

  return (
    // eslint-disable-next-line react/jsx-no-useless-fragment
    <>{children}</>
  );
};
```

Использование

```tsx
import { useTranslation } from "react-i18next";
import { memo, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { classNames } from "@/shared/lib/classNames/classNames";
import {
  DynamicModuleLoader,
  ReducersList,
} from "@/shared/lib/components/DynamicModuleLoader/DynamicModuleLoader";
import { useInitialEffect } from "@/shared/lib/hooks/useInitialEffect/useInitialEffect";
import { useAppDispatch } from "@/shared/lib/hooks/useAppDispatch/useAppDispatch";
import { Page } from "@/widgets/Page";
import { ArticleInfiniteList } from "../ArticleInfiniteList/ArticleInfiniteList";
import { ArticlesPageFilters } from "../ArticlesPageFilters/ArticlesPageFilters";
import { fetchNextArticlesPage } from "../../model/services/fetchNextArticlesPage/fetchNextArticlesPage";
import { initArticlesPage } from "../../model/services/initArticlesPage/initArticlesPage";
import { articlesPageReducer } from "../../model/slices/articlesPageSlice";
import cls from "./ArticlesPage.module.scss";
import { ArticlePageGreeting } from "@/features/articlePageGreeting";
import { ToggleFeatures } from "@/shared/lib/features";
import { StickyContentLayout } from "@/shared/layouts/StickyContentLayout";
import { ViewSelectorContainer } from "../ViewSelectorContainer/ViewSelectorContainer";
import { FiltersContainer } from "../FiltersContainer/FiltersContainer";

interface ArticlesPageProps {
  className?: string;
}

const reducers: ReducersList = {
  articlesPage: articlesPageReducer,
};

const ArticlesPage = (props: ArticlesPageProps) => {
  const { className } = props;
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [searchParams] = useSearchParams();

  const onLoadNextPart = useCallback(() => {
    dispatch(fetchNextArticlesPage());
  }, [dispatch]);

  useInitialEffect(() => {
    dispatch(initArticlesPage(searchParams));
  });

  const content = (
    <ToggleFeatures
      feature="isAppRedesigned"
      on={
        <StickyContentLayout
          left={<ViewSelectorContainer />}
          right={<FiltersContainer />}
          content={
            <Page
              data-testid="ArticlesPage"
              onScrollEnd={onLoadNextPart}
              className={classNames(cls.ArticlesPageRedesigned, {}, [
                className,
              ])}
            >
              <ArticleInfiniteList className={cls.list} />
              <ArticlePageGreeting />
            </Page>
          }
        />
      }
      off={
        <Page
          data-testid="ArticlesPage"
          onScrollEnd={onLoadNextPart}
          className={classNames(cls.ArticlesPage, {}, [className])}
        >
          <ArticlesPageFilters />
          <ArticleInfiniteList className={cls.list} />
          <ArticlePageGreeting />
        </Page>
      }
    />
  );

  return (
    <DynamicModuleLoader reducers={reducers} removeAfterUnmount={false}>
      {content}
    </DynamicModuleLoader>
  );
};

export default memo(ArticlesPage);
```

где articlesPageReducer

```tsx
import {
  createEntityAdapter,
  createSlice,
  PayloadAction,
} from "@reduxjs/toolkit";
import { StateSchema } from "@/app/providers/StoreProvider";
import {
  Article,
  ArticleType,
  ArticleView,
  ArticleSortField,
} from "@/entities/Article";
import { ARTICLES_VIEW_LOCALSTORAGE_KEY } from "@/shared/const/localstorage";
import { SortOrder } from "@/shared/types/sort";
import { ArticlesPageSchema } from "../types/articlesPageSchema";
import { fetchArticlesList } from "../../model/services/fetchArticlesList/fetchArticlesList";

const articlesAdapter = createEntityAdapter<Article>({
  selectId: (article) => article.id,
});

// получение структурированы статей
export const getArticles = articlesAdapter.getSelectors<StateSchema>(
  (state) => state.articlesPage || articlesAdapter.getInitialState()
);

const articlesPageSlice = createSlice({
  name: "articlesPageSlice",
  initialState: articlesAdapter.getInitialState<ArticlesPageSchema>({
    isLoading: false,
    error: undefined,
    ids: [],
    entities: {},
    view: ArticleView.SMALL,
    page: 1,
    hasMore: true,
    _inited: false,
    limit: 9,
    sort: ArticleSortField.CREATED,
    search: "",
    order: "asc",
    type: ArticleType.ALL,
  }),
  reducers: {
    setView: (state, action: PayloadAction<ArticleView>) => {
      state.view = action.payload;
      localStorage.setItem(ARTICLES_VIEW_LOCALSTORAGE_KEY, action.payload);
    },
    setPage: (state, action: PayloadAction<number>) => {
      state.page = action.payload;
    },
    setOrder: (state, action: PayloadAction<SortOrder>) => {
      state.order = action.payload;
    },
    setSort: (state, action: PayloadAction<ArticleSortField>) => {
      state.sort = action.payload;
    },
    setType: (state, action: PayloadAction<ArticleType>) => {
      state.type = action.payload;
    },
    setSearch: (state, action: PayloadAction<string>) => {
      state.search = action.payload;
    },
    initState: (state) => {
      const view = localStorage.getItem(
        ARTICLES_VIEW_LOCALSTORAGE_KEY
      ) as ArticleView;
      state.view = view;
      state.limit = view === ArticleView.BIG ? 4 : 9;
      state._inited = true;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchArticlesList.pending, (state, action) => {
        state.error = undefined;
        state.isLoading = true;

        if (action.meta.arg.replace) {
          articlesAdapter.removeAll(state);
        }
      })
      .addCase(fetchArticlesList.fulfilled, (state, action) => {
        state.isLoading = false;
        state.hasMore = action.payload.length >= state.limit;

        if (action.meta.arg.replace) {
          articlesAdapter.setAll(state, action.payload);
        } else {
          articlesAdapter.addMany(state, action.payload);
        }
      })
      .addCase(fetchArticlesList.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
      });
  },
});

export const { reducer: articlesPageReducer, actions: articlesPageActions } =
  articlesPageSlice;
```

## ULBI. Стилизация

### Темы

#### Тема. Контекст

```tsx
import React, { ReactNode, useEffect, useMemo, useState } from "react";
import { ThemeContext } from "../../../../shared/lib/context/ThemeContext";
import { Theme } from "@/shared/const/theme";
import { LOCAL_STORAGE_THEME_KEY } from "@/shared/const/localStorage";

interface ThemeProviderProps {
  initialTheme?: Theme;
  children: ReactNode;
}

// export enum Theme {
//     LIGHT = 'app_light_theme',
//     DARK = 'app_dark_theme',
//     ORANGE = 'app_orange_theme',
// }

const fallbackTheme = localStorage.getItem(LOCAL_STORAGE_THEME_KEY) as Theme;

const ThemeProvider = (props: ThemeProviderProps) => {
  const { initialTheme, children } = props;
  const [isThemeInitialized, setThemeInitialized] = useState(false);

  const [theme, setTheme] = useState<Theme>(
    initialTheme || fallbackTheme || Theme.LIGHT
  );

  useEffect(() => {
    if (!isThemeInitialized && initialTheme) {
      setTheme(initialTheme);
      setThemeInitialized(true);
    }
  }, [initialTheme, isThemeInitialized]);

  useEffect(() => {
    // Установка темы производится с помощью вставки класса для body
    document.body.className = theme;
    localStorage.setItem(LOCAL_STORAGE_THEME_KEY, theme);
  }, [theme]);

  const defaultProps = useMemo(
    () => ({
      theme,
      setTheme,
    }),
    [theme]
  );

  return (
    <ThemeContext.Provider value={defaultProps}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;
```

тема по умолчанию

```scss
:root {
  --bg-color: #e8e8ea;
  --inverted-bg-color: #090949;
  --primary-color: #0232c2;
  --secondary-color: #0449e0;
  --inverted-primary-color: #04ff04;
  --inverted-secondary-color: #049604;

  // redesigned
  --dark-bg-redesigned: #ffffff;
  --bg-redesigned: #eff5f6;
  --light-bg-redesigned: #e2eef1;
  --text-redesigned: #141c1f;
  --hint-redesigned: #adbcc0;
  --cancel-redesigned: #ff7777;
  --save-redesigned: #62de85;
  --icon-redesigned: #5ed3f3;
  --accent-redesigned: #00c8ff;

  // skeleton
  --skeleton-color: #fff;
  --skeleton-shadow: rgba(0 0 0 / 20%);

  // code
  --code-bg: #fff;

  // card
  --card-bg: #d5d5d7;

  // listbox
  --listbox-bg: #d5d5d7;
}
```

темная тема

```scss
.app_dark_theme {
  --bg-color: #090949;
  --inverted-bg-color: #e8e8ea;
  --primary-color: #04ff04;
  --secondary-color: #049604;
  --inverted-primary-color: #0232c2;
  --inverted-secondary-color: #0452ff;

  // redesigned
  --dark-bg-redesigned: #090f11;
  --bg-redesigned: #0c1214;
  --light-bg-redesigned: #151c1f;
  --text-redesigned: #dbdbdb;
  --hint-redesigned: #555555;
  --cancel-redesigned: #d95757;
  --save-redesigned: #6cd98b;
  --icon-redesigned: #74a2b2;
  --accent-redesigned: #5ed3f3;

  // skeleton
  --skeleton-color: #1515ad;
  --skeleton-shadow: #2b2be8;

  // code
  --code-bg: #1212a1;

  // card
  --card-bg: #0d0d6b;

  // listbox
  --listbox-bg: #0d0d6b;
}
```

Дополнительная, третья тема

```scss
.app_orange_theme {
  --bg-color: #faf4fb;
  --inverted-bg-color: #bd5012;
  --primary-color: #9a1a0e;
  --secondary-color: #d01f0e;
  --inverted-primary-color: #dbd5dc;
  --inverted-secondary-color: #faf4fb;

  // redesigned
  --dark-bg-redesigned: #fff3d6;
  --bg-redesigned: #f0c048;
  --light-bg-redesigned: #f2d791;
  --text-redesigned: #1b1311;
  --hint-redesigned: #b8b2a2;
  --cancel-redesigned: #ff5e5e;
  --save-redesigned: #52fa81;
  --icon-redesigned: #4875f0;
  --accent-redesigned: #1d59ff;

  // skeleton
  --skeleton-color: #fff;
  --skeleton-shadow: rgba(0 0 0 / 20%);

  // code
  --code-bg: #fff;

  // card
  --card-bg: #d5d5d7;

  // listbox
  --listbox-bg: #d5d5d7;
}
```
