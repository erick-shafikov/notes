```tsx
import React, { useCallback, useMemo, useRef } from "react";
import { View, Text, StyleSheet } from "react-native";
import BottomSheet, { BottomSheetView } from "@gorhom/bottom-sheet";

const App = () => {
  // ref
  const bottomSheetRef = useRef<BottomSheet>(null);

  // callbacks
  const handleSheetChanges = useCallback((index: number) => {
    console.log("handleSheetChanges", index);
  }, []);

  // renders
  return (
    <View _style={styles.container}>
      <BottomSheet ref={bottomSheetRef} onChange={handleSheetChanges}>
        <BottomSheetView _style={styles.contentContainer}>
          <Text>Awesome 🎉</Text>
        </BottomSheetView>
      </BottomSheet>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    backgroundColor: "grey",
  },
  contentContainer: {
    flex: 1,
    alignItems: "center",
  },
});

export default App;
```

# props

```ts
type BottomSheetProps = {
  index: number; //0 изначальный индекс; можно передать -1 для того что бы инициализировать bs скрытым
  snapPoints: number[]; //степень открытия
  snapPoints: [200; 500]; // в пикселях
  snapPoints:['100%']; // в %
  napPoints: [200; '50%']; // совмещенный вариант
  overDragResistanceFactor: 2.5; //настройки движения
  detached: false; // привязан bs к низу
  enableContentPanningGesture: true; //Включите взаимодействие с жестами панорамирования контента.
  enableHandlePanningGesture: true;
  enableOverDrag: true; //перетаскивание bs
  enablePanDownToClose false; //отключает закрытие на смахивание вниз
  enableDynamicSizing: false;
  animateOnMount: true;
};
```

# Стили:

- style - на весь контейнер
- backgroundStyle - для заднего фона
- handleStyle - стили ручки смахивания
- handleIndicatorStyle - индикатор ручки смахивания

# Настройки макета

- handleHeight - 24 высота ручки, рассчитывается автоматически
- containerHeight - 0 высота контейнера, рассчитывается автоматически
- contentHeight
- containerOffset
- topInset
- bottomInset
- maxDynamicContentSize

Настройки клавиатуры

```ts
type KeyboardConfiguration = {
  keyboardBehavior: "interactive" | "extend" | "fillParent"; // как будет вести себя Bs - по размеру клавиатуры, до макс snapPoint,
  keyboardBlurBehavior: "none" | "restore";
};
```

Коллбеки

```ts
type onChange = (index: number) => void;
type onAnimate = (fromIndex: number, toIndex: number) => void;
```

# Компоненты:

- handleComponent
- backdropComponent
- backgroundComponent
- footerComponent
- children - Scrollable

# Методы

```ts
// свернуть до индекса в snapIndex
type snapToIndex = (
  index: number,
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;

//свернуть до пикселя
type snapToPosition = (
  position: number,
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;

// развернуть
type expand = (
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;

// свернуть
type collapse = (
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;

// закрыть
type close = (
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;

// закрыть
type forceClose = (
  animationConfigs?: Animated.WithSpringConfig | Animated.WithTimingConfig
) => void;
```

# хуки

useBottomSheet
useBottomSheetDynamicSnapPoints
useBottomSheetSpringConfigs
useBottomSheetTimingConfigs

# доп компоненты:

- BottomSheetBackdrop
- BottomSheetFlatList
- BottomSheetFooter
- BottomSheetSectionList
- BottomSheetScrollView
- BottomSheetTextInput
- BottomSheetVirtualizedList
- BottomSheetView
