# VirtualizedList

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
