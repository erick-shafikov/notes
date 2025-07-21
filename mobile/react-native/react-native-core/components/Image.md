# Image

Компонент картинки

```js
<Image
  alt={String}
  blurRadius={Number}//для закругления
  crossOrigin={'anonymous' | 'use-credentials'}//crossOrigin настройки
  defaultSource={require('./img/default-pic.png')}// при ошибке загрузки
  fadeDuration={300}
  height={Number}
  loadingIndicatorSource={type ImageSource} //отображаемый контент при загрузке
  onError={({nativeEvent: {error} }) => void}
  onLayout={({nativeEvent: LayoutEvent}) => void}
  onLoad={({nativeEvent: ImageLoadEvent}) => void}
  onLoadEnd={() => void}
  onLoadStart={() => void}
  onProgress={({nativeEvent: {loaded, total} }) => void}
  resizeMethod={'auto' | 'resize' | 'scale'}
  referrerPolicy={'no-referrer' | 'no-referrer-when-downgrade' | 'origin' | 'origin-when-cross-origin' | 'same-origin' | 'strict-origin' | 'strict-origin-when-cross-origin' | 'unsafe-url'}//политика загрузки
  resizeMode={'cover' | 'contain' | 'stretch' | 'repeat' | 'center'}//растягивание изображений
  source={type ImageSource}
  src={String}
  srcSet={'https://reactnativedev/img/tiny_logopng 1x, https://reactnativedev/img/header_logosvg 2x'}
  tintColor={String}
  width={Number}
  // ANDROID
  progressiveRenderingEnabled={false} //флаг отвечающий за стримминг jpeg
  // IOS
  capInsets={} //
  // STYLE
  style={ImageStyleProps, LayoutProps, ShadowProps, Transforms}
/>
```

```js
// для статики
<Image source={require('./img/check.png')} />

// для сторонних ресурсов
<Image
  source={{ uri: "https://reactjs.org/logo-og.png@2x" }} // 2x - dpi
  style={{ width: 400, height: 400 }}
/>
```

При опционально цепочке загрузки

```js
// GOOD
<Image source={require("./my-icon.png")} />;

// BAD
const icon = this.props.active ? "my-icon-active" : "my-icon-inactive";
<Image source={require("./" + icon + ".png")} />;

// GOOD
const icon = this.props.active
  ? require("./my-icon-active.png")
  : require("./my-icon-inactive.png");
<Image source={icon} />;
```

методы

```js
abortPrefetch(requestId: Number) // requestId - число возвращаемое prefetch
getSize(uri: string,
  success: (width: number, height: number) => void,
  failure?: (error: any) => void,)
getSizeWithHeaders(uri: string,
  headers: {[index: string]: string},
  success: (width: number, height: number) => void,
  failure?: (error: any) => void,)
prefetch(url, callback)
queryCache(  urls: string[]): Promise<Record<string, 'memory' | 'disk' | 'disk/memory'>>
resolveAssetSource(source: ImageSourcePropType): {
  height: number;
  width: number;
  scale: number;
  uri: string;
};
```

```js
// типизация ImageSource
type ImageSource {
  uri: string,
  width: number,
  height: number,
  scale: number, //DPI
  method: string, //GET по умолчанию
  headers: object,
  body: string,
  // iOS only props
  bundleiOS: String,
  cacheiOS: 'default' | 'reload' | 'force-cache' | 'only-if-cached'

}
```
