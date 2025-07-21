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
