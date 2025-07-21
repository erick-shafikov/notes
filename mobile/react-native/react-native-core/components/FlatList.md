# FlatList

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
