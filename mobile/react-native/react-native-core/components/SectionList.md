# SectionList

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
