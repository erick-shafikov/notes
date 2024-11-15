- react компоненты переводятся в нативные компоненты
- логика воспроизводится в js

Фабрика - это система рендера React. Которая помогает решить такие проблемы как: синхронность, интеграция React Suspense, React concurrent фишки. Весь код компилирует в C++ код. Фазы: Render (js производит shadow tree которое преобразуется в С++), Commit (стадия изменения, формируется новое дерево), Mount (преобразуется в host view tree)

- Render: при создании элемента с помощью React Element вызывается React Shadow Node. Это случает только для React Host Component но не для React Composite Component.
- Commit: при изменении вызывается Yoga, которая позволяет произвести вычисления. Tree Promotion - стадия на которой, новой дерево отрисовывается. Операция асинхронная. Новое дерево отрисуется на следующий тик
- Mount: React Shadow Tree превращается в Host View Tree, отображая пиксели. Tree Diffing - стадия вычисления разницы, Tree Promotion - добавление разницы, View Mounting- отрисовка

Схема работы кросс-платформы

<img src='./assets/react-native/cross-plat.png' height=200 width=300/>

Используются View Flattening - для работы со вложенными компонентами

Headless JS - это то как запускаются задачи в JS, пока приложение свернуто

# Запуск проекта

- с помощью expo или sdk

<!--Компоненты--------------------------------------------------------------------------------------------------------------------->

# Компоненты

## ActivityIndicator

вращающийся спиннер спиннер

```js
import { ActivityIndicator } from "react-native";

const App = () => (
  <ActivityIndicator
    animating={true} //крутится или нет
    color={"color"}
    hidesWhenStopped={true} //(IOS) отображать или нет пока крутится
    size={"large" | "small"} //размер
  />
);
```

## Alert

Диалоговое окно с предупреждением. На ios можно любое количество кнопок. Для Android две кнопки OK и Cancel

```js
Alert.alert(
  title: String, //текст заголовка
  message: String, //сообщение
  buttons: type AlertButton[], //кнопки снизу от сообщения
  options: type AlertOption //конфигурация
);

Alert.prompt({
  title: String, //титул диалогового окна
  message: String, //Сообщение перед текстовым вводом
  callbackOptions: (text: String) => void | AlertButton[],
  type: 'default' | 'plain-text' | 'secure-text' | 'login-password' //IOS тип ввода
  defaultValue: String, //текст по умолчанию в поле ввода
  keyboardType: String //вариант раскладки клавиатуры
})

type AlertButtonStyle =  'default' | 'cancel' | 'destructive'

type AlertButton = {
  text: String;
  onPress: Function;
  style: 'default' | 'cancel' | 'destructive' //(IOS) вид кнопки
  isPreferred: false //(IOS) подчеркнутость кнопки
}

type AlertOption = {
  cancelable: false; //(Android) закрытие при клике
  userInterfaceStyle: 'light' | 'dark';
  onDismiss: Function //(Android) функция, которая срабатывает на закрытие
}
```

## Appearance

Позволяет узнать настройки телефона

```js
import { Appearance } from "react-native";

const colorScheme = Appearance.getColorScheme();
if (colorScheme === "dark") {
  // Use dark color scheme
}

//методы

getColorScheme(); // вернет 'light' | 'dark' | null;
setColorScheme("light" | "dark" | null);
addChangeListener(
  listener: (preferences: {colorScheme: 'light' | 'dark' | null}) => void,
): NativeEventSubscription;
```

## AppRegistry

для низкоуровневого взаимодействия

## AppState

```js
import { AppState } from "react-native";

const AppStateExample = () => {
  const appState = useRef(AppState.currentState);
  const [appStateVisible, setAppStateVisible] = useState(appState.current);

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (nextAppState) => {
      if (
        appState.current.match(/inactive|background/) &&
        nextAppState === "active"
      ) {
        console.log("App has come to the foreground!");
      }

      appState.current = nextAppState;
      setAppStateVisible(appState.current);
      console.log("AppState", appState.current);
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return (
    <View styles={styles.container}>
      <Text>Current state is: {appStateVisible}</Text>
    </View>
  );
};
```

События на подписку: change, memoryWarning, focus , blur. Метод один - addEventListener, метод - currentState

## Button

```jsx
<Button
  onPress={({nativeEvent: PressEvent})=>void} //(required) коллбек на нажатие
  title={'string'} //(required) текст
  color={"hexColor"} //цвет кнопки '#2196F3'  для Android и '#007AFF' IOS
  disabled={true} //активность кнопки
  touchSoundDisabled={false} //(Android) при нажатии воспроизводить звук
/>
```

## Dimensions

Dimensions - объект, который возвращает параметры экрана. Используется при определении размеров, которые требуются при первоначальной инициализации компонента

```js
//узнаем ширину экрана возвращает { width: number; height: number; scale: number; fontScale: number; }
const deviceWidth = Dimensions.get("window").width;
//использование
height: deviceWidth < 380 ? 160 : 300,
```

методы

```js
addEventListener(type: 'change', handler: ({window, screen} : DimensionValue) => void) //при изменении
get(dim: 'window' | 'screen') : { //метод на получение информации
  width: Number,
  height: Number,
  scale: Number,
  fontScale: Number
}
```

## FlatList

оптимизированный лист, data - желательно что бы был объект с ключом key

Советы по улучшению:

- использовать простые компоненты
- использовать memo, getItemLayout, keyExtractor
- избегать анонимных функций
- при прокрутке невидимые элементы уничтожаются

```js
<FlatList
  data={dataItems}// данные в виде массива
  renderItem={(// элемент доступен по itemData item
    item,  // элемент списка
    index: Number, //индекс в массиве
    separators: {
      highlight: () => void,
      unhighlight: () => void,
      updateProps: (select: "leading" | "trailing", newProps: any) => void,
    }
  ) => {
    return (
      <FlatLIstItem
        //элементы доступны itemDat . item
        text={itemData.item.text}
        key={item.key}
        onPress={() => this._onPress(item)}
        onShowUnderlay={separators.highlight}
        onHideUnderlay={separators.unhighlight}
      />
    );
  }}
  ItemSeparatorComponent={<Component />}// компонент, который будет между элементами
  ListEmptyComponent={<Component />} //если лист пустой
  ListFooterComponent={<Component />} //компонент внизу листа
  ListHeaderComponent={<Component />} //компонент заголовка

  ListFooterComponentStyle={{}} //стиль ListFooterComponent
  ListHeaderComponentStyle={{}} //стиль ListHeaderComponent
  columnWrapperStyle={{}} //numColumns > 1
  extraData={} // для оптимизации, если более 100 элементов, если внутренние компоненты завязаны на сторонних пропсах
  getItemLayout={(data, index) => {length: number, offset: number, index: number}} //функция которая будет возвращать размеры компонентов
  horizontal={false}
  initialNumToRender={Number}//сколько элементов отрендерить изначально
  initialScrollIndex={Number} //не работает без getItemLayout
  inverted={Boolean}
  keyExtractor={(item, index) => { // что бы не указывать key каждому элементу, можно использовать  keyExtractor, который извлечет
    return item.id; // получает коллбек с двумя параметрами item - элемент, index - индекс в массиве itemData
  }}
  numColumns={Number}// количество колонок
  onRefresh={Function}
  onEndReached={Function} //сработает при достижения нижней части списка с учетом onEndReachedThreshold
  onEndReachedThreshold={Number}// как далеко от конца списка сработает onEndReached
  onViewableItemsChanged={(callback: {changed: ViewToken[], viewableItems: ViewToken[]} => void)}// для корректного отображения индикатора загрузки
  progressViewOffset={Number} //расстояние до компонента загрузки
  refreshing={Boolean} // при загрузке контента установить в true
  removeClippedSubviews={true} //оптимизационный флаг
  scrollEnabled={Boolean} //Когда находится в ScrollView, должно быть true, по умолчанию false
  viewabilityConfig={
    minimumViewTime: Number, //количество мс сколько должно пройти что бы элемент стал доступен
    viewAreaCoveragePercentThreshold: Number, //количество процентов заполнения view что бы считалось что заполнено
    itemVisiblePercentThreshold: Number, // относительно родителя
    waitForInteraction: Boolean //флаг для определения включать ли прокрутку
  }

  ItemSeparatorComponent={
    Platform.OS !== "android" &&
    (({ highlighted }) => (
      <View style={[style.separator, highlighted && { marginLeft: 0 }]} />
    ))
  }
  />

```

Методы

```jsx
flashScrollIndicators(); //покажет скролл индикатор
getNativeScrollRef(); //доступ к компоненту скролла
getScrollResponder(); //компонент прокрутки
getScrollableNode();
scrollToEnd(); //вниз
scrollToIndex({
  index: Number,
  animated?: Boolean,
  viewOffset?: Number,
  viewPosition?: Number,
})
scrollToItem({
  animated?: ?Boolean,
  item: Item,
  viewPosition?: Number,
})
scrollToOffset({
  offset: Number;
  animated?: Boolean;
})
```

## Image

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

## ImageBackground

```js
<ImageBackground
  source={...}
  imageRef={}//ссылка на внутренний объект изображения
  imageStyle={{}} //Объект стилей
  style={{width: '100%', height: '100%'}}
/>
```

## Keyboard

Работа с клавиатурой осуществляется с помощью событийной модели

```js
const Example = () => {
  const [keyboardStatus, setKeyboardStatus] = useState("");

  useEffect(() => {
    const showSubscription = Keyboard.addListener("keyboardDidShow", () => {
      setKeyboardStatus("Keyboard Shown");
    });
    const hideSubscription = Keyboard.addListener("keyboardDidHide", () => {
      setKeyboardStatus("Keyboard Hidden");
    });

    return () => {
      showSubscription.remove();
      hideSubscription.remove();
    };
  }, []);

  return (
    <View styles={style.container}>
      <TextInput
        styles={style.input}
        placeholder="Click here…"
        onSubmitEditing={Keyboard.dismiss}
      />
      <Text styles={style.status}>{keyboardStatus}</Text>
    </View>
  );
};
```

Методы

```js
addListener( eventType:
| 'keyboardWillShow'
| 'keyboardDidShow'
| 'keyboardWillHide'
| 'keyboardDidHide'
| 'keyboardWillChangeFrame'
| 'keyboardDidChangeFrame');
scheduleLayoutAnimation(event: KeyboardEvent)
dismiss();
isVisible();
metrics();
```

## Modal

```js
<Modal
  animationType={"none" | "slide" | "fade"} // slide - выезжает снизу, fade -гаснет
  hardwareAccelerated={false} //(Android) аппаратное ускорение
  onDismiss={Function} //(IOS) функция на закрытие
  onOrientationChange={Function} //(IOS) сработает при смене ориентации
  onRequestClose={Function} //сработает при нажатии кнопки назад
  onShow={Function} //функция которая будет вызвана один раз при открытие
  presentationStyle={
    "fullScreen" | "pageSheet" | "formSheet" | "overFullScreen"
  } //(IOS)
  statusBarTranslucent={false} //(Android) должно ли диалоговое окно уходить
  supportedOrientations={
    "portrait" |
    "portrait-upside-down" |
    "landscape" |
    "landscape-left" |
    "landscape-right"
  } //(IOS)
  transparent={false} //прозрачный фон если true
  visible={true} //флаг отвечающий за видимость
/>
```

```js
const ModalButton = () => {
  const [modalVisible, setModalVisible] = useState(false);
  return (
    <View>
      //контейнер модального окна
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => {
          Alert.alert("Modal has been closed.");
          setModalVisible(!modalVisible);
        }}
      >
        // содержимое модального окна
        <View>
          <Text>Hello World!</Text>
          // Кнопка для закрытия
          <Pressable onPress={() => setModalVisible(!modalVisible)}>
            <Text>Hide Modal</Text>
          </Pressable>
        </View>
      </Modal>
      // Кнопка для открытия
      <Pressable onPress={() => setModalVisible(true)}>
        <Text>Show Modal</Text>
      </Pressable>
    </View>
  );
};
```

## Platform

**свойства**

- OS: 'ios' | 'android' | 'native' | 'default'
- Version

```js
import { Platform, StyleSheet } from "react-native";

const styles = StyleSheet.create({
  height: Platform.OS === "ios" ? 200 : 100,
  //альтернативный вариант через Platform.select borderWidth: Platform.OS === "android" ? 2 : 0,
  borderWidth: Platform.select({ iso: 0, android: 2 }),
});
```

```js
//разные компоненты для разных платформ, select позволяет вернуть то значение, которую нужно будет для ios или android
const Component = Platform.select({
  ios: () => require("ComponentIOS"),
  android: () => require("ComponentAndroid"),
})();
<Component />;
```

## Pressable

Обертка для объектов для того, что бы их сделать кликабельными

```js
<Pressable
  android_ripple={{//(Android) стилизация при нажатии на андроиде
    color:'color',
    borderless: 'boolean'
    radius: 'number'
    foreground: 'boolean'
  }}
  android_disableSound={false}
  unstable_pressDelay={Number}// настройка для onLongPress
  delayLongPress={Number}
  disabled={false}
  hitSlop={{ // дистанция от элемента, при которой сработает
    bottom: 20,
    left: null,
    right: undefined,
    top: 50
}}
// для hover web версии
  onHoverIn={({ nativeEvent: MouseEvent }) => undefined}
  onHoverOut={({ nativeEvent: MouseEvent }) => undefined} // если держать палец больше 500 мс
  onLongPress={({nativeEvent: PressEvent}) => undefined} // onPressIn -> onPressOut
  onPress={({nativeEvent: PressEvent}) => undefined} //вызывается при нажатии
  onPressIn={({nativeEvent: PressEvent}) => undefined} //вызывается при отпускании
  onPressOut={({nativeEvent: PressEvent}) => undefined}// как далеко считать onPressOut
  pressRetentionOffset={{bottom: 30, left: 20, right: 20, top: 20,}}
  _style={({ pressed }) => ViewStyle} //стиль может принимать функцию, которая принимает параметр pressed
  // пример
  _style={({ pressed }) => [
    styles.button,
    pressed ? styles.buttonPressed : null,
  ]}
 >
```

## RefreshControl

используется для жеста вниз на самом верху (обновление)

```js
<RefreshControl
  colors={String[]} //(Android) цвет индикатора
  enabled={true} //(Android) состояние обновления
  onRefresh={Function} // коллбек на срабатывание
  progressBackgroundColor={type Color} //(Android)
  progressViewOffset={Number} //отступ от верха
  refreshing={Boolean!} // индикатор
  size={'default', 'large'}//(Android)
  tintColor={type Color}//(IOS)
  title={String}//(IOS)
  titleColor={type Color}//(IOS)
/>
```

```js
const RefreshControl = () => {
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  return (
    <SafeAreaView>
      <ScrollView
        //  ScrollView принимает проп refreshControl
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <Text>Pull down to see RefreshControl indicator</Text>
      </ScrollView>
    </SafeAreaView>
  );
};
```

## ScrollView

Позволяет сделать контейнер прокручиваемым. Либо должен иметь фиксированную высоту, либо flex === 1. другие компоненты не предоставляют возможность для прокрутки

```js
//наследует от View
<ScrollView
StickyHeaderComponent={<Component />} // компонент который будет оставаться при прокрутке на месте
contentContainerStyle={type StyleView} // стиль обертки
contentInset={}
contentOffset={{x: 0, y: 0}} // стартовая позиция прокрутки
decelerationRate={'fast'|'normal'} // как быстро будет производится прокрутка
disableIntervalMomentum={false} //
disableScrollViewPanResponder={false} //
horizontal={false} // ориентация
invertStickyHeaders={false} // элемент внизу
keyboardDismissMode={'none'| 'on-drag' | 'interactive'} // режим скрытия клавиатура
keyboardShouldPersistTaps={'always' | 'never' | 'handled' | false | true} // режим скрытия клавиатуры
maintainVisibleContentPosition={{minIndexForVisible: Number, autoscrollToTopThreshold: Number}} //
onContentSizeChange={(contentWidth, contentHeight) => void} //При изменении размеров
onMomentumScrollBegin={Function} // Начало скролла
onMomentumScrollEnd={Function}// Окончание скролла
onScroll={(nativeEvent: { // Срабатывает на каждый фрейм
    contentInset: {bottom, left, right, top},
    contentOffset: {x, y},
    contentSize: {height, width},
    layoutMeasurement: {height, width},
    zoomScale
  })=>void}
onScrollBeginDrag={Function} // При срабатывании перетаскивания
onScrollEndDrag={Function}
pagingEnabled={Boolean} // Дискретная прокрутка
refreshControl={type Element} // ## RefreshControl
removeClippedSubViews={false} // убрать невидимые элементы
scrollEnabled={true} // возможность прокрутки
scrollEventThrottle={0} // тротлинг
showsHorizontalScrollIndicator={true}// показывать индикатор
showsVerticalScrollIndicator={true} //показывать горизонтальный индикатор
snapToAlignment={'start' | 'center' | 'end'} //связь привязки с прокруткой
snapToEnd={true} //snapToOffsets
snapToInterval={0}
snapToOffsets={Number[]}
snapToStart={Boolean}
stickyHeaderHiddenOnScroll={false} //при скролле stickyHeader будет скрываться
stickyHeaderIndices={Number[]} //какой из элементов будет прикрепляться к верху

//IOS only
alwaysBounceHorizontal={Boolean} // true - прокрутку подпрыгивает, defaults: false, если vertical === true
alwaysBounceVertical={true} // true - прокрутку подпрыгивает
automaticallyAdjustContentInsets={false} //корректировка при появлении клавиатуры
automaticallyAdjustKeyboardInsets={true} //Контролирует автоматическое добавление прокрутки
automaticallyAdjustsScrollIndicatorInsets={true} //Учет панели навигации
bounces={Boolean} //подпрыгивание при окончании прокрутки
bouncesZoom={Boolean} //ограничения при зуме
canCancelContentTouches={true} //отключение прокрутки
centerContent={false} //при малом количестве контента центрировать его
contentInset={{top: 0, left: 0, bottom: 0, right: 0}} //кула вставится контент
contentInsetAdjustmentBehavior={'automatic' | 'scrollableAxes' | 'never' | 'always'} //взаимодействие с safeArea
directionalLockEnabled={false} //блок при смене положения телефона
indicatorStyle={'default' | 'black' | 'white'} // цвет прокрутки
maximumZoomScale={1.0} // зум
minimumZoomScale={1.0} // зум
onScrollToTop={Function}
pinchGestureEnabled={true} // позволяет использовать жесты сжатия для увеличения и уменьшения масштаба
scrollIndicatorInsets={{ // насколько далеко
  top: number,
  left: number,
  bottom: number,
  right: number
}}
scrollToOverflowEnabled={Boolean}
scrollsToTop={true}//прокручивать верх при касании по статус бару
snapToAlignment={'start' | 'center' | 'end'}
zoomScale={1.0} //зум элементов scroll view

//Android only
endFillColor={'color'} // заполняет цветом пространство
fadingEdgeLength={0} //сколько элементов затемнить
nestedScrollEnabled={false} // вложенный скролл
overScrollMode={'auto' | 'always' | 'never'}
persistentScrollbar={false} // прозрачный при отсутствии
scrollPerfTag={String} // специальный тег для скролла
>
```

методы

```js
flashScrollIndicators(); //отобразить линию прокрутки

scrollTo( //прокрутка с плавной анимацией
  options?: {
    x?: number,
    y?: number,
    animated?: boolean
  } | number,
  deprecatedX?: number,
  deprecatedAnimated?: boolean,
  )

scrollToEnd(options?: {animated?: boolean})//скролл к низу
```

<!-- SectionList -->

## SectionList

Список разделенный секциями

- прокрученные элементы списка на сохраняются
- это PureComponent
- контент подгружается синхронно

```jsx
<SectionList
  renderItems={type RenderItems}
  sections={type Sections[]}
  extraData={type any}//проп, который при изменении будет вызывать перерендер
  initialNumToRender={10} //начальное количество элементов на отображение
  inverted={false} //в обратном порядке
  ItemSeparatorComponent={<Component />} //элемент разделителя
  keyExtractor={KeyExtractor} //достать ключи
  ListEmptyComponent={<Component />} //если лист пустой
  ListFooterComponent={<Component />} //Футер
  ListHeaderComponent={<Component />} //Хедер
  onRefresh={Function} //если есть данный проп то будет добавлена возможно "Потяните вверх для обновления" среагирует на вверх от списка на 100px
  onViewableItemsChanged={type OnViewableItemsChanged} //вызывается при изменении в количестве рядов
  refreshing={false}//индикатор обновления контента
  removeClippedSubviews={false} //для улучшения перформанса
  renderSectionFooter={type RenderSectionFooter} //элемент для конца каждой секции
  renderSectionHeader={type RenderSectionHeader} //элемент на начало каждой секции
  SectionSeparatorComponent={<Component />} //компонент разделителя1
  stickySectionHeadersEnabled={false} //для IOS === true фиксированный элемент предыдущей секции при прокрутки

/>
```

Типизация пропсов

```ts
type RenderItems = {
  item: Object, //(!) элемент данных на отображения
  index: number,
  section: Object, //объект секции
  separators: {
    highlight: () => void,
    unhighlight: () => void,
    updateProps: (select : 'leading' | 'trailing', newProps: Object) => void,
  },
}
type KeyExtractor = (item: object, index: number) => string
type OnViewableItemsChanged = (callback: {changed: ViewToken[], viewableItems: ViewToken[]}) => void
type RenderSectionFooter = ({section: type Section}) => Element ｜ null
type RenderSectionHeader = (info: {section: type Section}) => Element ｜ null

type Section = any;
```

Методы

```js
recordInteraction(); // заставляет пересчитать
scrollToLocation(params: {
  animated: Boolean,
  itemIndex: Number,
  sectionIndex: Number,
  viewOffset: Number,
  viewPosition: Number,
});
// IOS
flashScrollIndicators(); //отображение динии скролла
```

```js
const DATA = [
  {
    title: "Main dishes",
    data: ["Pizza", "Burger", "Risotto"],
  },
  {
    title: "Sides",
    data: ["French Fries", "Onion Rings", "Fried Shrimps"],
  },
  {
    title: "Drinks",
    data: ["Water", "Coke", "Beer"],
  },
  {
    title: "Desserts",
    data: ["Cheese Cake", "Ice Cream"],
  },
];

const App = () => (
  <SafeAreaView>
    <SectionList
      sections={DATA}
      keyExtractor={(item, index) => item + index}
      renderItem={({ item }) => (
        <View>
          <Text>{item}</Text>
        </View>
      )}
      renderSectionHeader={({ section: { title } }) => <Text>{title}</Text>}
    />
  </SafeAreaView>
);
```

<!-- StatusBAr -------------------------------------------------------------------------->

## StatusBar

Компонент статус бара

```js
<StatusBar
  animated={false}
  barStyle={"default" | "light-content" | "dark-content"} //стиль
  hidden={false}
  //Android only props
  currentHeight={Number}
  backgroundColor={"black"}
  translucent={false}
  //IOS only props
  networkActivityIndicatorVisible={false} //отображение индикатора сети
  showHideTransition={"fade" | "slide" | "none"} //анимация скрытия
/>
```

Методы

```js
popStackEntry(entry: StatusBarProps); //удаление стека
pushStackEntry();
replaceStackEntry();
setBackgroundColor();
setBarStyle();
setHidden();
setNetworkActivityIndicatorVisible() //(iOS)
setTranslucent(); //(Android)
```

## Switch

```js
// inherit ViewProps
<Switch
  disabled={false} //индикатор активности
  onChange={Function} //кб на изменение
  onValueChange={Function} //кб на изменение получает новое значение как аргумент
  thumbColor="color"//цвет переключателя
  trackColor={{false: type Color, true: type Color}}//
  value={Boolean} //значение

  // IOS only
  ios_backgroundColor={type Color}
/>
```

## Text

- в него может быть вложен только другой <Text />
- поддерживает интерактивность
- Inline элементы
- рекомендуется создать текстовый компонент для всего

```js
<Text
//Accessibility
accessibilityHint={String}
accessibilityLanguage={String}
accessibilityLabel={String}
accessibilityRole={type AccessibilityRole}
accessibilityState={type AccessibilityState}
accessibilityState={Array}
onAccessibilityAction={Function}
accessible={true} //активация Accessibility
adjustsFontSizeToFit={false} //автоматическое уменьшение стиля
allowFontScaling={true} //масштабирование текста
ellipsizeMode={'head'| 'middle'| 'tail'| 'clip'} //работает в паре с numberOfLines определяет как декорировать окончание строки
id={String}
maxFontSizeMultiplier={Number} // если allowFontScaling  === true определяет максимальное увеличение текста
minimumFontScale={Number} //
nativeID={String}
numberOfLines={0}
onLayout={({nativeEvent: LayoutEvent}) => void} //вызывается на монтирование и изменение
onLongPress={({nativeEvent: PressEvent}) => void}
onMoveShouldSetResponder={({nativeEvent: PressEvent}) => boolean}
onPress={({nativeEvent: PressEvent}) => void} //  onPress = onPressIn + onPressOut
onPressIn={({nativeEvent: PressEvent}) => void}
onPressOut={({nativeEvent: PressEvent}) => void}
//обработка движений
onResponderGrant={({nativeEvent: PressEvent}) => void ｜ boolean}
onResponderMove={({nativeEvent: PressEvent}) => void}
onResponderRelease={({nativeEvent: PressEvent}) => void}
onResponderTerminate={({nativeEvent: PressEvent}) => void}
onResponderTerminationRequest={({nativeEvent: PressEvent}) => boolean}
onStartShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}

onTextLayout={(TextLayoutEvent) => mixed}
pressRetentionOffset={type Rect | Number}
role={type Role} //
selectable={false}//позволяет выделить текст, по умолчанию текст выделять нельзя
_style={} //
testID={} //
//aria props
aria-busy={false}
aria-disabled={false}
aria-expanded={false}
aria-label={String}
aria-selected={Boolean}
//Android only props
android_hyphenationFrequency={'none' | 'normal' | 'full'} //расстановка переносов
dataDetectorType={'none' | 'phoneNumber' | 'link' | 'email' | 'all'} //превратить в интерактивный текст
disabled={false} //
selectionColor ={type Color} //цвет выделения текста
textBreakStrategy ={'simple'|'highQuality'|'balanced'}
//iOS props only
dynamicTypeRamp={'caption2'|'caption1'|'footnote'|'subheadline'|'callout'|'body'|'headline'|'title3'|'title2'|'title1'|'largeTitle'}
/>
suppressHighlighting={false} //при нажатии на  текст
lineBreakStrategyIOS={'none'|'standard'|'hangul-word'|'push-out'}
```

## TextInput

**пропсы**

- onChangeText - коллбек на изменение текста
- value дял контролируемого значения
- maxLength - максимальное количество символов

```js
<TextInput
onChangeText={(text:string) => void}
allowFontScaling={true} //Указывает, должны ли шрифты масштабироваться в соответствии с настройками специальных возможностей размера текста
autoCapitalize={'none' | 'sentences' | 'words' | 'characters'} //режим заглавной буквы
autoComplete={'additional-name'|'address-line1'|'address-line2'|'birthdate-day'|'birthdate-full'|'birthdate-month'|'birthdate-year'|'cc-csc'|'cc-exp'|'cc-exp-day'|'cc-exp-month'|'cc-exp-year'|'cc-number'|'country'|'current-password'|'email'|'family-name'|'given-name'|'honorific-prefix'|'honorific-suffix'|'name'|'new-password'|'off'|'one-time-code'|'postal-code'|'street-address'|'tel'|'username'|'cc-family-name'|'cc-given-name'|'cc-middle-name'|'cc-name'|'cc-type'|'nickname'|'organization'|'organization-title'|'url'|'gender'|'name-family'|'name-given'|'name-middle'|'name-middle-initial'|'name-prefix'|'name-suffix'|'password'|'password-new'|'postal-address'|'postal-address-country'|'postal-address-extended'|'postal-address-extended-postal-code'|'postal-address-locality'|'postal-address-region'|'sms-otp'|'tel-country-code'|'tel-device'|'tel-national'|'username-new'}
autoCorrect={true} // автокоррекция componentDidMount или useEffect
autoFocus={false} // автофокус при componentDidMount
blurOnSubmit={true} // потеря фокуса при окончании ввода текста
caretHidden={false} // скрытие каретки
contextMenuHidden={false} // скрытие меню
defaultValue={String}
editable={true}
enterKeyHint={'enter' | 'done' | 'next' | 'previous' | 'search' | 'send'}
inputMode={'decimal' | 'email' | 'none' | 'numeric' | 'search' | 'tel' | 'text' | 'url'}
keyboardType={'default'|'email-address'|'numeric'|'phone-pad'|'ascii-capable'|'numbers-and-punctuation'|'url'|'number-pad'|'name-phone-pad'|'decimal-pad'|'twitter'|'web-search'|'visible-password'} //тип клавиатуры под input
maxFontSizeMultiplier={0} // если allowFontScaling  === true то определяет число максимального увеличения шрифта
maxLength={0}
multiline={false}
onBlur={Function} // коллбек на потерю фокуса
onChange={({nativeEvent: {eventCount, target, text}}) => void} // коллбек на изменение
onChangeText={Function} // функция на изменение текста
onContentSizeChange={({nativeEvent: {contentSize: {width, height} }}) => void}
onEndEditing={Function} // коллбек на окончание ввода
//
onPressIn={({nativeEvent: PressEvent}) => void}
onPressOut={({nativeEvent: PressEvent}) => void}
onFocus={({nativeEvent: LayoutEvent}) => void}
onKeyPress={({nativeEvent: {key: keyValue} }) => void}
onLayout={({nativeEvent: LayoutEvent}) => void} // вызывается при монтировании компонента
onScroll={({nativeEvent: {contentOffset: {x, y} }}) => void}
onSelectionChange={({nativeEvent: {selection: {start, end} }}) => void}
onSubmitEditing={({nativeEvent: {text, eventCount, target}}) => void}
placeholder={'string'}
placeholderTextColor={'hexString'}
readOnly={false}
returnKeyType={'done'|'go'|'next'|'search'|'send'|'none'|'previous'|'default'|'emergency-call'|'google'|'join'|'route'|'yahoo'}
secureTextEntry={false}
selection={{start: number,end: number}}
selectionColor={'hexString'}
selectTextOnFocus={true}
showSoftInputOnFocus={true}
textAlign={'left'|'center'|'right'}
textContentType ={'none'|'addressCity'|'addressCityAndState'|'addressState'|'birthdate'|'birthdateDay'|'birthdateMonth'|'birthdateYear'|'countryName'|'creditCardExpiration'|'creditCardExpirationMonth'|'creditCardExpirationYear'|'creditCardFamilyName'|'creditCardGivenName'|'creditCardMiddleName'|'creditCardName'|'creditCardNumber'|'creditCardSecurityCode'|'creditCardType'|'emailAddress'|'familyName'|'fullStreetAddress'|'givenName'|'jobTitle'|'location'|'middleName'|'name'|'namePrefix'|'nameSuffix'|'newPassword'|'nickname'|'oneTimeCode'|'organizationName'|'password'|'postalCode'|'streetAddressLine1'|'streetAddressLine2'|'sublocality'|'telephoneNumber'|'URL'|'username'}

//Android props only
cursorColor={type Color} //цвет каретки
disableFullscreenUI={false}
importantForAutofill={'auto' | 'no' | 'noExcludeDescendants' | 'yes' | 'yesExcludeDescendants'}//
inputAccessoryViewID={String}
numberOfLines={0}
returnKeyLabel={String}
rows={0} //
textBreakStrategy={'simple'|'highQuality'|'balanced'}//перенос текста
underlineColorAndroid={type Color}

//iOS props only
clearButtonMode={'never'| 'while-editing'| 'unless-editing'| 'always'} //только для текст полей в одну линию
clearTextOnFocus ={true}//удаляет текст при наведении
contextMenuHidden={false} //скрывать контекстное меню
enablesReturnKeyAutomatically={false}//скрыть кнопке возврата
dataDetectorTypes={'phoneNumber', 'link', 'address', 'calendarEvent', 'none', 'all'} //интерактивность текста
inlineImageLeft={String} //изображение должно быть в /android/app/src/main/res/drawable inlineImageLeft='search_icon'
inlineImagePadding={0}//расстояние от inlineImageLeft
keyboardAppearance={'default'| 'light'| 'dark'} //цвет клавиатуры ввода
rejectResponderTermination={true}//передача событий выше
scrollEnabled={true}
spellCheck={true}
textContentType={}
passwordRules={String}
lineBreakStrategyIOS={'none', 'standard', 'hangul-word', 'push-out'}
// не поддерживается borderLeftWidth borderTopWidth borderRightWidth, borderBottomWidth, borderTopLeftRadius, borderTopRightRadius,
// borderBottomRightRadius, borderBottomLeftRadius,
style={type Style}
/>
```

методы

```js
.focus()
.blur()
clear()
isFocused()
```

## TouchableHighlight

компонент который реагирует на нажатие

## TouchableOpacity

компонент который реагирует на нажатие

## TouchableWithoutFeedback

- поддерживает только одного ребенка

```js
function MyComponent(props: MyComponentProps) {
  return (
    <View {...props} _style={{ flex: 1, backgroundColor: "#fff" }}>
      <Text>My Component</Text>
    </View>
  );
}

<TouchableWithoutFeedback onPress={() => alert("Pressed!")}>
  <MyComponent />
</TouchableWithoutFeedback>;
```

## View

основной контейнер - элемент
берет столько сколько можно

```js
<View

  accessible={true}//доступны ли для касания
  hitSlop={{ top: 10, bottom: 10, left: 0, right: 0 }}//насколько далеко от элемента будут срабатывать события касания
  onLayout={({nativeEvent: LayoutEvent}) => void}// вызовется, когда будут определены размеры

  // при движении пальца
  onMoveShouldSetResponder={({nativeEvent: PressEvent}) => boolean}// срабатывает на каждое касание
  onMoveShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}// если нужно предотвращение при движении от родителя, функция должна возвращать true
  onResponderGrant={({nativeEvent: PressEvent}) => void ｜ boolean}
  onResponderMove={({nativeEvent: PressEvent}) => void}
  onResponderReject={({nativeEvent: PressEvent}) => void}
  onResponderRelease={({nativeEvent: PressEvent}) => void}
  onResponderTerminate={({nativeEvent: PressEvent}) => void}
  onResponderTerminationRequest={({nativeEvent: PressEvent}) => void}
  onStartShouldSetResponder={({nativeEvent: PressEvent}) => boolean}
  onStartShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}
  //
  pointerEvents={'box-none'|'none'|'box-only'|'auto'}

  //Android only props
  tabIndex={0 | -1}
  //Style
  style={type ViewStyle}
/>
```

## VirtualizedList

базовый компонент для списков. Использовать в случае для большей кастомизации. Виртуальные списки создаю в видимых частях элементы и удаляют в невидимых

- не меняет состояние
- PureComponent
- При быстром скролле могут образовать пустые промежутки, так как элементы создаются асинхронно
- использовать key в паре с keyExtractor

```js
//наследуется от ScrollViewProps
<VirtualizedList

data={type any} //данные передаваемые в getItem и getItemCount
getItem={(data: Any, index: number) => any}
getItemCount={(data: any) => number}
renderItem={type ReactElement} // принимает data prop
CellRendererComponent={type ReactElement} //обертка над каждым компонентом
ItemSeparatorComponent={type ReactElement} //будет находится между каждым элементом
//компоненты и стилистка заголовка и подвала
ListEmptyComponent={type ReactElement}
ListFooterComponent={type ReactElement}
ListFooterComponentStyle={type ViewStyleProps}
ListHeaderComponent={type ReactElement}
ListHeaderComponentStyle={type ViewStyleProps}
debug={false} //дополнительные данные для дебага
extraData={any} //специальный проп для вызова рендера при его изменения
getItemLayout={(
  data: any,
  index: number,
) => {length: number, offset: number, index: number}}
horizontal={false}
initialNumToRender={10}
initialScrollIndex={0}
inverted={false}
keyExtractor={(item: any, index: Number) => string}
maxToRenderPerBatch={0} //максимальное количество элементов для рендеринга
onEndReached={(info: {distanceFromEnd: number}) => void}
onEndReachedThreshold={2} //на каком элементе с конца должен срабатывать onEndReached
onRefresh={Function} // функция, которая будет срабатывать на "потяните вверх чтобы обновить"
onScrollToIndexFailed={(info: {
  index: number,
  highestMeasuredFrameIndex: number,
  averageItemLength: number,
}) => void} //кб на неудачный переход к элементу списка
onStartReached={
TYPE
(info: {distanceFromStart: number}) => void} //вызывается один раз когда доходит до onStartReachedThreshold
onStartReachedThreshold={2} //на каком элементе с начала сработает onStartReached
onViewableItemsChanged={(callback: {changed: ViewToken[], viewableItems: ViewToken[]}) => void} //срабатывает при изменении видимости рядов
persistentScrollbar={false}
progressViewOffset={Number} //для корректного отображения индикатора загрузки
refreshControl={type Element} // переопределит <RefreshControl> компонент
refreshing={false} //индикатор загрузки
removeClippedSubviews={false}
renderScrollComponent={(props: object) => element} //элемент скролла
viewabilityConfig={ViewabilityConfig}
viewabilityConfigCallbackPairs={}
updateCellsBatchingPeriod={Number}
windowSize={21} //при значении 21 будет отрисовано 10 экранов вверх и 10 экранов вниз
/>
```

методы

```js
flashScrollIndicators();
getScrollableNode();
getScrollRef();
getScrollResponder();
scrollToEnd(params?: {animated?: boolean});
scrollToIndex(params: { //прокрутка до элемента с индексом
  index: number;
  animated?: boolean;
  viewOffset?: number;
  viewPosition?: number;
});
scrollToItem(); //прокрутка до элемента
scrollToOffset(params: { //прокрутка до определенного места в списке
  offset: number;
  animated?: boolean;
});
```

## Типизация аргументов

### PressEvent

```js
type PressEvent = {};
```

<!--Стилизация--------------------------------------------------------------------------------------------------------------------->

# Стилизация

## Выбор платформы

Можно назвать файлы компонентов как Component.android.js и Component.ios.js но в импортировать там, где нужно как Component

## StyleSheet - объект, который создает классы

методы

compose - позволяет объединить два объекта стиля, конфликты переопределяются

```js
const styles1 = StyleSheet.create({
  container: {},
  text: {},
});
const styles2 = StyleSheet.create({
  container: {},
  text: {},
});

const container = StyleSheet.compose(styles1.container, styles2.container);
const text = StyleSheet.compose(styles1.text, styles2.text);
```

**create** - позволяет создать объект стилей

**flatten** - объединяет стили в один плоский объект принимает в виде объекта

```js
const flattenStyle = StyleSheet.flatten([page.text, typography.header]);
```

**absoluteFillObject** - шорткат для позиционирования

```js
absoluteFillObject = {
  position: "absolute",
  top: 0,
  left: 0,
  bottom: 0,
  right: 0,
};
// применение
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  box3: {
    ...StyleSheet.absoluteFillObject,
    top: 0,
    left: 0,
    width: 100,
    height: 100,
    backgroundColor: "green",
  },
});
```

**hairlineWidth**

создает сплошную горизонтальную линию

```js
const styles = StyleSheet.create({
  __TESTING__: {
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});
```

## Image style props

```js
const ImageStyleProps = {
  backfaceVisibility: "visible" | "hidden",
  backgroundColor: Color,
  borderBottomLeftRadius: Number,
  // аналогично borderBottomRightRadius
  borderColor: Color,
  borderRadius: Number,
  //borderTopLeftRadius,  borderTopRightRadius
  borderWidth: Number,
  opacity: 1,
  overflow: "visible" | "hidden",
  objectFit: "cover" | "contain" | "fill" | "scale-down",
  resizeMode: "cover" | "contain" | "stretch" | "repeat" | "center",
  tintColor: Color,
  //Android only props
  overlayColor: Color,
};
```

## Layout props

Пропсы для контейнера

```js
const LayoutProps = {
  alignContent:
    "flex-start" |
    "flex-end" |
    "center" |
    "stretch" |
    "space-between" |
    "space-around" |
    "space-evenly",
  alignItems: "flex-start" | "flex-end" | "center" | "stretch" | "baseline",
  alignSelf: "flex-start" | "flex-end" | "center" | "stretch" | "baseline",
  aspectRatio: Number | String,
  borderBottomWidth: Number,
  //аналогично borderEndWidth, borderLeftWidth, borderRightWidth, borderStartWidth, borderTopWidth, borderWidth
  bottom: Number, //расстояние внизу от контейнера
  columnGap: Number,
  direction: "inherit" | "ltr" | "rtl",
  display: "none" | "flex",
  // flexDirection  === column, alignContent  === flex-start, flexShrink === 0
  flex: Number, // 1 - весь контейнер, 0 - размерность по width и height, -1 - по minWidth и minHeight
  flexBasis: Number,
  flexDirection: "row" | "row-reverse" | "column" | "column-reverse",
  flexGrow: Number,
  flexShrink: Number,
  flexWrap: "wrap" | "nowrap" | "wrap-reverse",
  gap: Number,
  height: Number, //может быть в процентах
  justifyContent:
    "flex-start" |
    "flex-end" |
    "center" |
    "space-between" |
    "space-around" |
    "space-evenly",
  left: Number, //расстояние от левого края компонента
  //аналогично height, start, end, top
  margin: Number,
  //аналогично marginBottom, marginEnd, marginHorizontal, marginLeft, marginRight, marginStart, marginTop, marginVertical
  maxHeight: Number,
  maxWidth: Number,
  minHeight: Number,
  minWidth: Number,
  overflow: "visible" | "hidden" | "scroll",
  padding: Number,
  //аналогично paddingBottom, paddingEnd, paddingHorizontal, paddingLeft, paddingRight, paddingStart, paddingTop, paddingVertical
  position: "absolute" | "relative",
  rowGap: Number,
  width: Number, //может быть в процентах
  zIndex: Number,
};
```

## Shadow props

```js
const ShadowProps = {
  elevation: Number,
  shadowColor: Color,
  // iOS only props
  shadowOffset: {
    width: Number,
    height: Number,
  },
  shadowOpacity: Number,
  shadowRadius: Number,
};
```

## Text style props

```js
const textProps = {
    color: Color,
    fontFamily: 'FontName',
    fontSize: number,
    fontWeight: 'normal'|'bold'|'100'|'200'|'300'|'400'|'500'|'600'|'700'|'800'|'900',
    fontVariant: 'small-caps'|'oldstyle-nums'|'lining-nums'|'tabular-nums'|'proportional-nums',
    letterSpacing: number,
    lineHeight: number,
    textAlign: 'auto'|'left'|'right'|'center'|'justify',
    textDecorationLine: 'none'|'underline'|'line-through'|'underline line-through',
    textDecorationStyle : 'solid'|'double'|'dotted'|'dashed',
    textShadowColor: 'color',
    textShadowOffset: {width?: number, height?: Number},
    textShadowRadius: 0,
    textTransform: 'none'|'uppercase'|'lowercase'|'capitalize',
    userSelect: 'auto', 'text', 'none', 'contain', 'all',

    //Android only
    includeFontPadding: true,
    textAlignVertical: 'auto' | 'top' | 'bottom' | 'center',
    verticalAlign: 'auto' | 'top' | 'bottom' | 'middle',

    //iOS only
    textDecorationColor: Color
    textDecorationStyle : 'solid' | 'double' | 'dotted' | 'dashed',
    writingDirection: 'auto' | 'ltr' | 'rtl'

  },
```

## View style props

Пропсы для элементов

```js
const viewStyleProps = {
  backfaceVisibility: "visible" | "hidden",
  backgroundColor: "color",
  borderBottomColor: "color",
  borderRadius: number, //borderBottomEndRadius, borderBottomLeftRadius, borderBottomRightRadius, borderBottomStartRadius, borderStartEndRadius, borderStartStartRadius, borderEndEndRadius, borderEndStartRadius, borderTopEndRadius,borderTopLeftRadius, borderTopRightRadius, borderTopStartRadius
  borderWidth: number, //borderBottomWidth, borderLeftWidth, borderRightWidth, borderTopWidth
  borderColor: "color", //borderEndColor, borderLeftColor, borderRightColor, borderStartColor, borderTopColor
  borderStyle: "solid" | "dotted" | "dashed",
  opacity: number,
  // определяет как будут обрабатываться касания
  // auto - касания обрабатываются
  // none - касания не обрабатываются
  // box-none - касания на view не обрабатываются, только на дочерних
  // box-only - касания обрабатываются только на view,но не на дочерних элементах
  pointerEvents: "auto" | "box-none" | "box-only" | "none",

  // android only
  elevation: Number,
  //iOS only
  borderCurve: "circular" | "continuous",
  cursor: "auto" | "pointer",
};
```

<!--Работа с анимацией------------------------------------------------------------------------------------------------------------->

# Работа с анимацией

Работа с анимацией происходит посредству библиотеке Animated. Общий подход

```js
// пример анимации opacity
const index = () => {
  // создаем ссылку на значение, которое будет изменяться, от какого значения начинаем
  const fadeAnim = useRef(new Animated.Value(0)).current;

  //функция анимации появления
  const fadeIn = () => {
    // Animated - библиотека, функция timing принимает ссылку и за 5000 мс доводит значение в current до 1
    Animated.timing(fadeAnim, {
      toValue: 1, // к какому значению придем
      duration: 5000,
      useNativeDriver: true,
    }).start();
  };

  const fadeOut = () => {
    // за 3 секунды доведет до 0
    Animated.timing(fadeAnim, {
      toValue: 0,
      duration: 3000,
      useNativeDriver: true,
    }).start();
  };

  return (
    <SafeAreaView>
      <Animated.View // Анимируемый объект View
        styles={[
          styles.fadingContainer,
          {
            opacity: fadeAnim, // ссылка на значение
          },
        ]}
      >
        <Text>Fading View!</Text>
      </Animated.View>
      <Button title="Fade In View" onPress={fadeIn} />
      <Button title="Fade Out View" onPress={fadeOut} />
    </SafeAreaView>
  );
};
```

База анимации - это величина, которую будем анимировать. То есть одно из значений стилей. Доступны для анимации:

```js
Animated.Value(0); //для скалярных.
Animated.ValueXY({ x: 0, y: 0 }); //для векторных величин.

//значение анимации передают ссылкой
const animationValue = useRef(new Animated.Value(0)).current;

//методы Value
animationValue.setValue(offset: Number); //установит значение напрямую, остановив анимацию
animationValue.setOffset(offset: Number); //установит смещение
animationValue.flattenOffset(); //объединение задержки
animationValue.extractOffset(); //сброс до начальных значений
animationValue.addListener(callback: (state: {value: number}) => void): string;
animationValue.removeListener(id: string);
animationValue.stopAnimation(callback?: (value: number) => void);
animationValue.resetAnimation(callback?: (value: number) => void);
animationValue.interpolate(config: { // InterpolationConfigType
    inputRange: number[];
    outputRange: number[] | string[];
    easing?: ((input: number) => number) | undefined;
    extrapolate?: ExtrapolateType | undefined; // ExtrapolateType = 'extend' | 'identity' | 'clamp'
    extrapolateLeft?: ExtrapolateType | undefined; // 'extend' | 'identity' | 'clamp'
    extrapolateRight?: ExtrapolateType | undefined; // 'extend' | 'identity' | 'clamp'
  }); //конфигурация для начальных и конечных значений
animationValue.animate(animation, callback);


//методы как у Value кроме getLayout
getLayout(): {left: Animated.Value, top: Animated.Value};
```

interpolation

```js
//интерполяция числовых значений
value.interpolate({
  inputRange: [0, 1],
  outputRange: [0, 100],
});

//применение значений
style={{
    opacity: this.state.fadeAnim, // Binds directly
    transform: [{
      translateY: this.state.fadeAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [150, 0]  // 0 : 150, 0.5 : 75, 1 : 0
      }),
    }],
  }}

//применение более двух значений значений
value.interpolate({
  inputRange: [-300, -100, 0, 100, 101],
  outputRange: [300, 0, 1, 0, 0],
});

//применение значений градусные меры
value.interpolate({
  inputRange: [0, 360],
  outputRange: ['0deg', '360deg'],
});
```

Далее нужно значение передать в функции задания и запуска анимации:

```js
const value = useRef(new Animated.Value(0)).current;


Animated.decay( //с замедлением под конец,
  value,
  config: {
    velocity: Number,
    deceleration: 0.997,
    isInteraction: true, //создает ли дескриптор взаимодействия
    useNativeDriver: true
});

Animated.timing(value, config : { //для использования easing функций
  duration: 500, //длительность в мс
  easing: type Easing, //кривая анимации
  delay: 0, //задержка
  isInteraction: true,
  useNativeDriver: true
});

Animated.spring(value, config: { //модель пружины,
  friction: 7,
  tension: 40
  speed: 12,
  bounciness: 8,
});
```

Запуск анимации

```js
Animated.timing({}).start(({ finished }) => {
  //выполнится по окончанию анимации
});

// Остановка анимации
Animated.stop();
Animated.reset();
```

Доступные для анимации элементы: Animated.Image, Animated.ScrollView, Animated.Text, Animated.View, Animated.FlatList, Animated.SectionList

```js
createAnimatedComponent(); // Animated.View
```

Анимации могут быть собраны в композицию с помощью

```js
Animated.delay(time: Number); //запуск анимации с задержкой

Animated.parallel(//параллельный запуск нескольких анимаций
  animations: type CompositeAnimation[],
  config?: ParallelConfig
);

Animated.sequence(animations: type CompositeAnimation[]); // запуск анимаций последовательно

Animated.stagger( //запускает анимации параллельно, но с задержкой
  time: Number,
  animations: type CompositeAnimation[]
);

Animated.loop(
  animation: type CompositeAnimation[],
  config?: LoopAnimationConfig
)
```

```js
Animated.sequence([
  Animated.decay(position, {
    velocity: { x: Number, y: Number },
    deceleration: 0.997,
    useNativeDriver: true,
  }),
  Animated.parallel([
    Animated.spring(position, {
      toValue: { x: 0, y: 0 }, // return to start
      useNativeDriver: true,
    }),
    Animated.timing(twirl, {
      toValue: 360,
      useNativeDriver: true,
    }),
  ]),
]).start();
```

Действия со значениями

```js
Animated.add(a: type Animated, b: type Animated); // +
Animated.subtract(a: type Animated, b: type Animated); // -
Animated.divide(a: type Animated, b: type Animated); // /
Animated.modulo(a: type Animated, modulus: Number); // /
Animated.multiply(a: type Animated, b: type Animated); // *
Animated.diffClamp(a: Animated, min: number, max: number); //в промежутке
```

Анимация обработки жестов происходит с помощью метода event

```js
event(
  argMapping: Mapping[], // type Mapping = {[key: string]: Mapping} | AnimatedValue;
  config?: EventConfig : {
    listener?: ((event: NativeSyntheticEvent<T>) => void) | undefined;
    useNativeDriver: boolean;
  } //
): (...args: any[]) => void;
```

```js
 onScroll={Animated.event(
   // scrollX = e.nativeEvent.contentOffset.x
   [{nativeEvent: {
        contentOffset: {
          x: scrollX
        }
      }
    }]
 )}

 //attachNativeEvent открепляет
```

```js
import React, { useRef } from "react";
import {
  SafeAreaView,
  ScrollView,
  Text,
  StyleSheet,
  View,
  ImageBackground,
  Animated,
  useWindowDimensions,
} from "react-native";

const images = new Array(6).fill(
  "https://images.unsplash.com/photo-1556740749-887f6717d7e4"
);

const App = () => {
  // значение анимации
  const scrollX = useRef(new Animated.Value(0)).current;
  // ширина окна
  const { width: windowWidth } = useWindowDimensions();

  return (
    <SafeAreaView>
      <View>
        <ScrollView
          horizontal={true}
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          // на скролл изменяем значение
          onScroll={Animated.event([
            {
              nativeEvent: {
                contentOffset: {
                  x: scrollX,
                },
              },
            },
          ])}
          scrollEventThrottle={1}
        >
          {images.map((image, imageIndex) => {
            return (
              <View key={imageIndex}>
                <ImageBackground source={{ uri: image }}>
                  <View>
                    <Text>{"Image - " + imageIndex}</Text>
                  </View>
                </ImageBackground>
              </View>
            );
          })}
        </ScrollView>
        <View>
          {images.map((image, imageIndex) => {
            //интерполяция относительно индекса и ширины окна
            const width = scrollX.interpolate({
              // три точки
              inputRange: [
                windowWidth * (imageIndex - 1),
                windowWidth * imageIndex,
                windowWidth * (imageIndex + 1),
              ],
              // три значения
              outputRange: [8, 16, 8],
              extrapolate: "clamp",
            });
            return (
              <Animated.View
                key={imageIndex}
                // применяется к ширине точки
                styles={[styles.normalDot, { width }]} //style
              />
            );
          })}
        </View>
      </View>
    </SafeAreaView>
  );
};
```

## Easing

```js
import React from "react";
import {
  Animated,
  Easing,
  SectionList,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import type { EasingFunction } from "react-native";

const App = () => {
  // значение
  let opacity = new Animated.Value(0);

  // функция коллбек на запуск по клику
  const animate = (easing: EasingFunction) => {
    // устанавливаем значение в 0
    opacity.setValue(0);
    // настройка анимации
    Animated.timing(opacity, {
      toValue: 1,
      duration: 1200,
      easing,
      useNativeDriver: true,
      // запуск
    }).start();
  };

  // значений настройки
  const size = opacity.interpolate({
    // значения входные
    inputRange: [0, 1],
    // значения для анимации с учетом функции
    outputRange: [0, 80],
  });

  // анимируемый стиль вынесен в отдельный массив
  const animatedStyles = [
    styles.box,
    {
      opacity,
      width: size, // от 0 до 80
      height: size, // от 0 до 80
    },
  ];

  return (
    <View>
      <StatusBar hidden={true} />
      <Text>Press rows below to preview the Easing!</Text>
      <View>
        <Animated.View styles={animatedStyles} /> //анимируемый объект
      </View>
      <SectionList
        sections={SECTIONS}
        keyExtractor={(item) => item.title}
        renderItem={({ item }) => (
          <TouchableOpacity onPress={() => animate(item.easing)}>
            <Text>{item.title}</Text>
          </TouchableOpacity>
        )}
        renderSectionHeader={({ section: { title } }) => <Text>{title}</Text>}
      />
    </View>
  );
};

const SECTIONS = [
  {
    title: "Predefined animations",
    data: [
      { title: "Bounce", easing: Easing.bounce },
      { title: "Ease", easing: Easing.ease },
      { title: "Elastic", easing: Easing.elastic(4) },
    ],
  },
  {
    title: "Standard functions",
    data: [
      { title: "Linear", easing: Easing.linear },
      { title: "Quad", easing: Easing.quad },
      { title: "Cubic", easing: Easing.cubic },
    ],
  },
  {
    title: "Additional functions",
    data: [
      {
        title: "Bezier",
        easing: Easing.bezier(0, 2, 1, -1),
      },
      { title: "Circle", easing: Easing.circle },
      { title: "Sin", easing: Easing.sin },
      { title: "Exp", easing: Easing.exp },
    ],
  },
  {
    title: "Combinations",
    data: [
      {
        title: "In + Bounce",
        easing: Easing.in(Easing.bounce),
      },
      {
        title: "Out + Exp",
        easing: Easing.out(Easing.exp),
      },
      {
        title: "InOut + Elastic",
        easing: Easing.inOut(Easing.elastic(1)),
      },
    ],
  },
];
```

## InteractionManager

Позволяет запланировать какие-либо действия после анимации или взаимодействия

```js
InteractionManager.runAfterInteractions(() => {
  // ...long-running synchronous task...
});
```

Альтернативы: requestAnimationFrame() - с течением времени, setImmediate/setTimeout(), runAfterInteractions() - вызывается после анимации или взаимодействия

## LayoutAnimation

Автоматически анимирует View в новом макете

```js
import React, { useState } from "react";
import { LayoutAnimation, Platform } from "react-native"; //импорт платформы

// активация для android
if (
  Platform.OS === "android" &&
  UIManager.setLayoutAnimationEnabledExperimental
) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}
const App = () => {
  const [expanded, setExpanded] = useState(false);

  return (
    <View>
      <TouchableOpacity
        onPress={() => {
          // подход - вызов пере изменением состояния
          LayoutAnimation.configureNext(LayoutAnimation.Presets.spring);
          // альтернативный вызов - вызов с конфигурацией
          LayoutAnimation.configureNext({
            duration: 500,
            create: { type: "linear", property: "opacity" },
            update: { type: "spring", springDamping: 0.4 },
            delete: { type: "linear", property: "opacity" },
          });
          setExpanded(!expanded);
        }}
      >
        <Text>Press me to {expanded ? "collapse" : "expand"}!</Text>
      </TouchableOpacity>
      {expanded && (
        <View>
          <Text>I disappear sometimes!</Text>
        </View>
      )}
    </View>
  );
};
```

Методы

```js
LayoutAnimation.configureNext( //анимация на следующий layout
  config: {
    duration: 0, //мс
    create: { // так же может быть update, delete
      type: type AnimationType,
      property: type LayoutProperty,
      springDamping : Number,
      initialVelocity : Number
      delay: Number,
      duration: Number
    }
  },
  onAnimationDidEnd?: () => void, //по окончанию анимации
  onAnimationDidFail?: () => void,); //при отмене анимации


  LayoutAnimation.create(duration, type, creationProp)
```

Типы анимации (поле type в create, delete, update): spring, linear, easeInEaseOut, easeIn, easeOut, keyboard

## PanResponder

Позволяет обработать несколько касаний в одно. PanResponder включает в себя InteractionManager, что бы блокировать поток. Основная задача обрабатывать жесты для анимации

```js
onPanResponderMove: (event: EventType, gestureState: GestureStateType) => {};

type GestureStateType = {
  stateID: String,
  moveX: Number,
  moveY: Number,
  x0: Number,
  y0: Number,
  dx: Number,
  dy: Number,
  vx: Number,
  vy: Number,
  numberActiveTouches: Number,
};

const ExampleComponent = () => {
  const panResponder = React.useRef(
    PanResponder.create({
      // Запрос на взаимодействие 1 стадия взаимодействия
      onStartShouldSetPanResponder: (evt, gestureState) => true, //будет ли view отвечать на касания
      onStartShouldSetPanResponderCapture: (evt, gestureState) => true,
      onMoveShouldSetPanResponder: (evt, gestureState) => true,
      onMoveShouldSetPanResponderCapture: (evt, gestureState) => true,

      onPanResponderReject: (e, gestureState) => {}

      onPanResponderGrant: (evt, gestureState) => {}, //старт на касание gestureState.d{x,y} === 0 в данный момент, можно подсветить элемент с которым идет взаимодействие
      onPanResponderStart: (e, gestureState) => {}
      onPanResponderEnd: (e, gestureState) => {}

      onPanResponderMove: (evt, gestureState) => {
        //срабатывает на движение
      },
      onPanResponderTerminationRequest: (evt, gestureState) => true,
      onPanResponderRelease: (evt, gestureState) => {}, //при окончании взаимодействия
      onPanResponderTerminate: (evt, gestureState) => {}, //другой компонент стал ответчиком
      onShouldBlockNativeResponder: (evt, gestureState) => {
        // Returns whether this component should block native components from becoming the JS
        // responder. Returns true by default. Is currently only supported on android.
        return true;
      },
    })
  ).current;

  // передать в компонент
  return <View {...panResponder.panHandlers} />;
};
```

```js
// пример с draggable
import React, { useRef } from "react";
import { Animated, View, StyleSheet, PanResponder, Text } from "react-native";

const App = () => {
  // значение в виде точки
  const pan = useRef(new Animated.ValueXY()).current;

  const panResponder = useRef(
    //создаем PanResponder
    PanResponder.create({
      //должен ли реагировать на жесты
      onMoveShouldSetPanResponder: () => true,
      // при начале движения
      onPanResponderMove: Animated.event([null, { dx: pan.x, dy: pan.y }]),
      // при окончании движения
      onPanResponderRelease: () => {
        pan.extractOffset();
      },
    })
  ).current;

  return (
    <View>
      <Text>Drag this box!</Text>
      <Animated.View
        styles={{
          transform: [{ translateX: pan.x }, { translateY: pan.y }],
        }}
        // передаем пропсы для движения
        {...panResponder.panHandlers}
      >
        <View />
      </Animated.View>
    </View>
  );
};
```

```js
// пример с draggable с возвратом на исходную позицию
import React, { useRef } from "react";
import { Animated, View, StyleSheet, PanResponder, Text } from "react-native";

const App = () => {
  const pan = useRef(new Animated.ValueXY()).current;
  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: Animated.event([null, { dx: pan.x, dy: pan.y }]),
      onPanResponderRelease: () => {
        Animated.spring(pan, {
          toValue: { x: 0, y: 0 },
          useNativeDriver: true,
        }).start();
      },
    })
  ).current;

  return (
    <View>
      <Text styles={styles.titleText}>Drag & Release this box!</Text>
      <Animated.View
        styles={{
          transform: [{ translateX: pan.x }, { translateY: pan.y }],
        }}
        {...panResponder.panHandlers}
      >
        <View />
      </Animated.View>
    </View>
  );
};
```

## Transforms

Изменение объектов

```js
const transform = {
  // В виде массива объектов трансформации
  transform: [{rotateX: '45deg'}, {rotateZ: '0.785398rad'}],
  // В виде строки
  transform: 'rotateX(45deg) rotateZ(0.785398rad)',
  //Все виды
  matrix: number[],
  perspective: number,
  //вращение
  rotate: string,
  rotateX: string,
  rotateY: string,
  rotateZ: string,
  //масштабирование
  scale: number,
  scaleX: number,
  scaleY: number,
  //перенос по осям
  translateX: number,
  translateY: number,
  //скашивание
  skewX: string,
  skewY: string
}
```

Transform Origin

точка поворота

```js
{
  transformOrigin: '20px',
  transformOrigin: 'bottom',
  transformOrigin: '10px 2px',
  transformOrigin: 'left top',
  transformOrigin: 'top right',
  transformOrigin: '2px 30% 10px',
  transformOrigin: 'right bottom 20px',
  // Using numeric values
  transformOrigin: [10, 30, 40],
  // Mixing numeric and percentage values
  transformOrigin: [10, '20%', 0],
}
```

<!--Обработка жестов-------------------------------------------------------------------------------------------------------------->

# Обработка жестов

## PressEvent

```js
{
  changedTouches: [PressEvent],
  identifier: 1,
  locationX: 8,
  locationY: 4.5,
  pageX: 24,
  pageY: 49.5,
  target: 1127,
  timestamp: 85131876.58868201,
  touches: []
}
```

<!--Хуки-------------------------------------------------------------------------------------------------------------->

# Хуки

## useWindowDimensions

- useWindowDimensions : ScaledSize Используется для измерений, которые нужно каждый рендер в отличает от Dimensions

```js
interface ScaledSize {
  width: number;
  height: number;
  scale: number;
  fontScale: number;
}
//в компоненте
const { width, height } = useWindowDimensions();
//контрольные измерения
const marginTopDistance = height < 380 ? 30 : 100;
//применение
<View style={[styles.rootContainer, { marginTop: marginTopDistance }]}>
```

## useColorScheme

```js
useColorScheme: "light" | "dart" | null; //- возвращает цветовую схему
```

```js
const App = () => {
  const colorScheme = useColorScheme();
  return (

  );
};
```
