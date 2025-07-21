# ScrollView

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
refreshControl={type Element} //
# RefreshControl
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
