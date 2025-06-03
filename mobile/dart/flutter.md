# Общие сведения

 - каждый виджет имеет метод build, который возвращает виджет
 - каждый верхнеуровневый виджет имеет ключ const Widget({super.key});

<!-------------------------------- Виджеты -------------------------------->

# Виджеты

**Appbar**

Виджет отвечающий за меню

```dart
//пример stateless виджета
class AppBar extends StatelessWidget {
  const AppBar({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // виджет appBar
      appBar: AppBar(
        // титул
        title: const Text('AppBar Demo'),
        // кнопки
        actions: <Widget>[
          // кнопка вызова SnackBar
          IconButton(
            icon: const Icon(Icons.add_alert),
            tooltip: 'Show Snackbar',
            onPressed: () {
              // по нажатию вызвать SnackBar
              ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('This is a snackbar')));
            },
          ),
          // кнопка перехода
          IconButton(
            icon: const Icon(Icons.navigate_next),
            tooltip: 'Go to the next page',
            onPressed: () {},
          ),
        ],
      ),
       notificationPredicate: (ScrollNotification notification) {
            return notification.depth == 1;
          },
      body: const Center(
        child: Text(
          'This is the home page',
          style: TextStyle(fontSize: 24),
        ),
      ),
      //нижняя часть appbar
      bottom: Widget[]
    );
  }
}
```

## Center ##

Виджет для центрирования элементов

```dart
 Center(
  child: Widget
  )
```

## Column ##
flex элементы вертикального позиционирования элементов
 - не скроллится
 - нельзя передавать ListView (ошибка неограниченной высоты), либо обернуть SizeBox()



```dart 
class _ColumnWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    throw const Column(
      children: <Widget>[
        Text('Deliver features faster'),
        Text('Craft beautiful UIs'),
        // займет все пространство
        Expanded(
          child: FittedBox(
            child: FlutterLogo(),
          ),
        ),
      ],
    );
  }
}
```

## Container
прямоугольный элемент предназначенный для рисования, позиционирования и изменения размеров виджета
Использует блочную модель

```dart
Container(
  decoration: BoxDecoration()
  child: Widget
)

```

## DefaultTabController
Виджет для построения табов

## FutureBuilder

```dart
import 'package:flutter/material.dart';

// формирование состояния
class FutureBuilderExampleApp extends StatelessWidget {//b-plate
  const FutureBuilderExampleApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: FutureBuilderExample(),
    );
  }
}
class FutureBuilderExample extends StatefulWidget {
  const FutureBuilderExample({super.key});

  @override
  State<FutureBuilderExample> createState() => _FutureBuilderExampleState();
}

// наследуется от FutureBuilderExample
class _FutureBuilderExampleState extends State<FutureBuilderExample> {
  //через 2 секунды значение _calculation изменится на 'Data loaded'
  final Future<String> _calculation = Future<String>.delayed(
    const Duration(seconds: 2),
    () => 'Data Loaded',
  );

  @override
  Widget build(BuildContext context) {
    return DefaultTextStyle(
      // оборачиваем в FutureBuilder
      child: FutureBuilder<String>(
        // разрешенное значение _calculation
        future: _calculation,
        //в snapshot лежит состояние String
        builder: (BuildContext context, AsyncSnapshot<String> snapshot) {
          List<Widget> children;
          // snapshot имеет флаг hasData
          if (snapshot.hasData) {
            children = <Widget>[
                // snapshot хранит состояние в поле data
                child: Text('Result: ${snapshot.data}'),
            ];
            // флаг для ошибки
          } else if (snapshot.hasError) {
            children = <Widget>[...];
          } else {
            children = const <Widget>[...];
          }
        },
      ),
    );
  }
}
```

## GridView ## 
- располагает по сетке
- виджет с возможностью скроллинга

GridView.count - позволяет задать количество колонок
GridView.extent - позволяет задать максимально количество ширины плитки

```dart
GridView.count(
  restorationId: 'grid_view_demo_grid_offset',
  crossAxisCount: 2,
  mainAxisSpacing: 8,
  crossAxisSpacing: 8,
  padding: const EdgeInsets.all(8),
  childAspectRatio: 1,
  children: _photos.map<Widget>((photo) {
    return _GridDemoPhotoItem(
      photo: photo,
      tileStyle: type,
      );
      }).toList(),
      ),
```

## Image ##

```dart


```


## ListTile ##

ListTile(
  leading: icon
  title: String
  subtitle: String
  trailing: icon
) - виджет - карточка


## ListView ## 

Виджет предоставляющий листы с возможностью скроллинга

```dart
Widget _buildList() {
  return ListView(
    scrollDirection: Axis.horizontal, //горизонтальная ориентация
    children: [...],
  );
}
```

## MaterialApp
Оборачивает приложения в Material библиотеку 

```dart MaterialApp(
  //свойство темы
  theme: ThemeData ()
)
```

## Row 
flex элементы для горизонтального принимает массив виджетов в children и располагает их горизонтально

```dart 
Row(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly, //mainAxisAlignment - координация объекта равномерно
  children: [
    const IconButton(
      ...
      ),
    Expanded(...), //child Expanded - виджета будет растянут, если все children обернуты в Expanded, то они будут распределены равномерно
    // у Expanded есть свойство flex:int - которое === flexBasis
    const IconButton(...),
  ],
)
```

## Stack ##
для позиционирования элементов слоями

## Stream Builder ##

```dart
import 'dart:async';
import 'package:flutter/material.dart';

class StreamBuilderExampleApp extends StatelessWidget {
  const StreamBuilderExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: StreamBuilderExample(),
    );
  }
}

class StreamBuilderExample extends StatefulWidget {
  const StreamBuilderExample({super.key});

  @override
  State<StreamBuilderExample> createState() => _StreamBuilderExampleState();
}

class _StreamBuilderExampleState extends State<StreamBuilderExample> {
  //в переменной _bids
  final Stream<int> _bids = (() {
    // controller - стрим
    late final StreamController<int> controller;
    // controller - принимает int в качестве сообщений
    controller = StreamController<int>(
      onListen: () async {
        await Future<void>.delayed(const Duration(seconds: 1));
        // controller.add добавляем событие
        controller.add(1);
        await Future<void>.delayed(const Duration(seconds: 1));
        await controller.close();
      },
    );
    return controller.stream;
  })();

  @override
  Widget build(BuildContext context) {
    return DefaultTextStyle(
      style: Theme.of(context).textTheme.displayMedium!,
      textAlign: TextAlign.center,
      child: Container(
        alignment: FractionalOffset.center,
        color: Colors.white,
        //child принимает StreamBuilder<тип данных на стриме>
        child: StreamBuilder<int>(
          // stream - _bids
          stream: _bids,
          // snapshot - асинхронное свойство
          builder: (BuildContext context, AsyncSnapshot<int> snapshot) {
            List<Widget> children;
            // обработка ошибки
            if (snapshot.hasError) {
              children = <Widget>[...];
            } else {
              // connectionState поле snapshot отвечающее за состояние стрима
              switch (snapshot.connectionState) {
                // ничего не происходит на стриме
                case ConnectionState.none:
                  children = const <Widget>[];
                // Режим ожидания
                case ConnectionState.waiting:
                  children = const <Widget>[];
                // Режим получения сообщения
                case ConnectionState.active:
                  children = <Widget>[
                    Padding(
                      padding: const EdgeInsets.only(top: 16),
                      // data - где находятся данные
                      child: Text('\$${snapshot.data}'),
                    ),
                  ];
                // закрытие стрима
                case ConnectionState.done:
                  children = <Widget>[
                    const Icon(
                      Icons.info,
                      color: Colors.blue,
                      size: 60,
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 16),
                      child: Text('\$${snapshot.data} (closed)'),
                    ),
                  ];
              }
            }

            return Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: children,
            );
          },
        ),
      ),
    );
  }
}
```

## Text ##
для стилизованного текста, первый аргумент String - текст
style - свойство для стилизации

```dart
Text('Hello world');
```


## runApp 
функция, которая запускает виджет внутри себя

**title** (String) - название приложения

**home** (Widget) - корневая страница

корневой виджет оборачивает в Material 

```dart
class MyScaffold extends StatelessWidget {
  const MyScaffold({super.key});

  @override
  Widget build(BuildContext context) {

    return Material(
      
      child: Column(
        ...
      ),
    );
  }
}
```
или Scaffold - базовый набор элементов

<!-------------------------------- Стилизация -------------------------------->


# Стилизация виджетов #

## BoxDecoration ## 

```dart
BoxDecoration(
  //цвет рамки
  border: Border
  borderRadius: const BorderRadius.all(Radius.circular(8)),
  color: Color 
  margin: const EdgeInsets.all(4),
)
```

## Expanded ## 

Обмачивает виджет для автоматического растягивания используется для Row, Column

```dart
Row(
  crossAxisAlignment: CrossAxisAlignment.center,
  children: [
    Expanded(
      child: Image.asset('images/pic1.jpg'),
    ),
    Expanded(
      child: Image.asset('images/pic2.jpg'),
    ),
    Expanded(
      child: Image.asset('images/pic3.jpg'),
    ),
  ],
);
```

```dart

***свойство flex***

Row(
  children: [
    Expanded(
      child: Image.asset('images/pic1.jpg'),
    ),
    // Разделит на 4 части для 3 элементов и 2 из 4 отдаст под данный элемент
    Expanded(
      flex: 2,
      child: Image.asset('images/pic2.jpg'),
    ),
    Expanded(
      child: Image.asset('images/pic3.jpg'),
    ),
  ],
);
```

## mainAxisAlignment ## 

Ориентация по главной оси

```dart
//пример для Row
Row(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
  children: [...],
);
//пример для Column
Column(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
  children: [...],
);

```

## mainAxisSize ## 

Предназначено для расзмеров

```dart
Row(
  //children будут ужаты до максимально возможного размера
  mainAxisSize: MainAxisSize.min,
  children: [...],
)
```

<!-------------------------------- Обработка жестов -------------------------------->

# theme #

```dart
MaterialApp(
  title: appName,
  theme: ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Colors.purple,
      brightness: Brightness.dark,
    ),W
    textTheme: TextTheme(
      displayLarge: const TextStyle(
        fontSize: 72,
        fontWeight: FontWeight.bold,
      ),
      titleLarge: GoogleFonts.oswald(
        fontSize: 30,
        fontStyle: FontStyle.italic,
      ),
      bodyMedium: GoogleFonts.merriweather(),
      displaySmall: GoogleFonts.pacifico(),
    ),
  ),
  home: const MyHomePage(
    title: appName,
  ),
);
```

<!-------------------------------- Обработка жестов -------------------------------->

# Обработка жестов #

GestureDetector - класс для обертки жестов

```dart 
Widget build(BuildContext context) {
    return GestureDetector(
      //cb для обработки жеста
      onTap: () {
        print('MyButton was tapped!');
      },
      //объект для взаимодействия
      child: Container(
        ...
      ),
    );
  }
```

**onTap**

Срабатывает при нажатии не элемент

**onPressed**

# Состояние приложения #

Во flutter компоненты наследуются от StatelessWidget или от StatefulWidget, данные классы предоставляют метод build для отображения виджетов

Пример счетчика
```dart
//объект c состоянием
class Counter extends StatefulWidget {
  const Counter({super.key});
//обязательный метод createState, который возвращает виджет
  @override
  State<Counter> createState() => _CounterState();
}

//класс компонента с состоянием
class _CounterState extends State<Counter> {
  //состояние
  int _counter = 0;

//функция, которая изменяет состояние
  void _increment() {
    //устанавливаем новое состояние
    setState(() {
      //cb на обработку состояния
      _counter++;
    });
  }

@override
  Widget build(BuildContext context) {
    
    return Row(
      children: <Widget>[
        ElevatedButton(
          onPressed: _increment,//обработка 
          child: const Text('Increment'),
        ),
        //достаем значение состояния
        Text('Count: $_counter'),
      ],
    );
  }

}
```

Stateful могут использовать Stateless

# Анимации

## AnimatedBuilder ##

```dart

// создать виджет с состоянием
class AnimatedBuilderExample extends StatefulWidget {
  const AnimatedBuilderExample({super.key});

  @override
  State<AnimatedBuilderExample> createState() => _AnimatedBuilderExampleState();
}

class _AnimatedBuilderExampleState extends State<AnimatedBuilderExample>
//добавить миксин анимации
    with TickerProviderStateMixin {
  // контроллер анимации
  late final AnimationController _controller = AnimationController(
    //настройки анимации
    duration: const Duration(seconds: 10),
    vsync: this,
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // использование AnimatedBuilder
    return AnimatedBuilder(
      // контроллер анимации
      animation: _controller,
      // анимируемый объект 
      child: Container(
        width: 200.0,
        height: 200.0,
        color: Colors.green,
        child: const Center(
          child: Text('Whee!'),
        ),
      ),
      //конструктор анимации
      builder: (BuildContext context, Widget? child) {
        
        return Transform.rotate(
          angle: _controller.value * 2.0 * math.pi,
          child: child,
        );
      },
    );
  }
}
```
<!-- Bloc --------------------------------------------------------------------------------------->


# Bloc # 
основывается на Stream 

**Cubit** - класс наследуемый от BlocBase, вызывает функции и изменяет состояние

```dart 
class CounterCubit extends Cubit<int> {
  //инициализация с заданным параметром
  CounterCubit() : super(0);
  //с параметром
  CounterCubit(int initialState) : super(initialState);
  //функция для изменения состояния
  void increment() => emit(state + 1);
  //функция которая будет срабатывать на каждое изменение
  @override
  void onChange(Change<int> change) {
    super.onChange(change);
    print(change);
  }

  //функция которая будет срабатывать при ошибке
  @override
  void onError(Object error, StackTrace stackTrace) {
    print('$error, $stackTrace');
    super.onError(error, stackTrace);
  }
}
```

Использование

```dart
void main() {
  final cubit = CounterCubit();
  print(cubit.state); // 0
  cubit.increment();
  print(cubit.state); // 1
  cubit.close();
}
```
С использованием Stream

```dart
Future<void> main() async {
  final cubit = CounterCubit();
  //добавляем слушателя на изменения 
  final subscription = cubit.stream.listen(print); // 1
  cubit.increment();
  await Future.delayed(Duration.zero);
  await subscription.cancel();
  await cubit.close();
}
```

**Bloc**

```dart
sealed class CounterEvent {}

final class CounterIncrementPressed extends CounterEvent {}

class CounterBloc extends Bloc<CounterEvent, int> {
  CounterBloc() : super(0) { 
    on<CounterIncrementPressed>((event, emit) {
      emit(state + 1);
    },
    transformer: debounce(const Duration(milliseconds: 300)),
    )
  };

  void onChange(Change<int> change) {
    super.onChange(change);
    print(change);
  }

  @override
  void onEvent(CounterEvent event) {
    super.onEvent(event);
    print(event);
  }

// onTransition позволяет отследить изменение, отобразив предыдущее состояние
  @override
  void onTransition(Transition<CounterEvent, int> transition) {
    super.onTransition(transition);
    print(transition);
  }
}

//использование

Future<void> main() async {
  final bloc = CounterBloc();
  print(bloc.state); // 0
  bloc.add(CounterIncrementPressed());
  await Future.delayed(Duration.zero);
  print(bloc.state); // 1
  await bloc.close();
}

или

Future<void> main() async {
  final bloc = CounterBloc();
  final subscription = bloc.stream.listen(print); // 1
  bloc.add(CounterIncrementPressed());
  await Future.delayed(Duration.zero);
  await subscription.cancel();
  await bloc.close();
}
```

Cubit - проще, так как в Bloc нужно объявлять обработчики события
Bloc - лучше отслеживает события, есть возможность добавить transformer

**BlocBuilder** 

виджет который принимает Bloc и builder функцию

```dart
BlocBuilder<BlocA, BlocAState>(
  bloc: blocA, // предоставляет локальный bloc
  builder: (context, state) {
    // возвращает виджет с состоянием 
  },
  buildWhen: (previousState, state) {
    //функция которая принимает предыдущее состояние и текущее, если эта функция вернет true, то build вызывается заново 
  },
);
```
**BlocSelector**

виджет, аналогичный BlocBuilder позволяет избежать повторных build-ов

```dart
BlocSelector<BlocA, BlocAState, SelectedState>(
  selector: (state) {
    // Отличие здесь, так как selector возвращает state, а buildWhen воз-т boolean
  },
  builder: (context, state) {
    // return widget here based on the selected state.
  },
);
```
**BlocProvider**

позволяет передать через контекст состояние BlocProvider.of<T>(context), позволяет передать bloc, cubit

```dart
BlocProvider(
  create: (BuildContext context) => BlocA(),
  lazy: Boolean // флаг который позволяет создавать провайдеры лениво
  child: ChildA(),
);
```
Использование

```dart
// with extensions
context.read<BlocA>();

// without extensions
BlocProvider.of<BlocA>(context);
```
**MultiBlocProvider**

позволяет работать со множественным контекстом и избежать вложенности

```dart
BlocProvider<BlocA>(
  create: (BuildContext context) => BlocA(),
  child: BlocProvider<BlocB>(
    create: (BuildContext context) => BlocB(),
    child: BlocProvider<BlocC>(
      create: (BuildContext context) => BlocC(),
      child: ChildA(),
    ),
  ),
);

MultiBlocProvider(
  providers: [
    BlocProvider<BlocA>(
      create: (BuildContext context) => BlocA(),
    ),
    BlocProvider<BlocB>(
      create: (BuildContext context) => BlocB(),
    ),
    BlocProvider<BlocC>(
      create: (BuildContext context) => BlocC(),
    ),
  ],
  child: ChildA(),
);
```

**BlocListener**

позволяет отслеживать изменения в bloc

```dart
BlocListener<BlocA, BlocAState>(
  bloc: blocA,
  listener: (context, state) {
    // do stuff here based on BlocA's state
  },
  child: const SizedBox(),
);

//MultiBlocListener

MultiBlocListener(
  listeners: [
    BlocListener<BlocA, BlocAState>(
      listener: (context, state) {},
    ),
    BlocListener<BlocB, BlocBState>(
      listener: (context, state) {},
    ),
    BlocListener<BlocC, BlocCState>(
      listener: (context, state) {},
    ),
  ],
  child: ChildA(),
);
```
**BlocConsumer**

Конечная точка, которая отрисовывает UI опираясь на bloc

```dart
BlocConsumer<BlocA, BlocAState>(
  listenWhen: (previous, current) {
    // return true/false to determine whether or not
    // to invoke listener with state
  },
  listener: (context, state) {
    // do stuff here based on BlocA's state
  },
  buildWhen: (previous, current) {
    // return true/false to determine whether or not
    // to rebuild the widget with state
  },
  builder: (context, state) {
    // return widget here based on BlocA's state
  },
);
```
**Применение BlocProvider**

```dart
//создаем sealed класс (нельзя импортировать, он абстрактный)
sealed class CounterEvent {}
final class CounterIncrementPressed extends CounterEvent {}
final class CounterDecrementPressed extends CounterEvent {}
//создаем bloc
class CounterBloc extends Bloc<CounterEvent, int> {
  //изначальное состояние === 0
  CounterBloc() : super(0) {
    //2 действия
    on<CounterIncrementPressed>((event, emit) => emit(state + 1));
    on<CounterDecrementPressed>((event, emit) => emit(state - 1));
  }
}

class CounterApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      //оборачиваем все в BlocProvider
      home: BlocProvider(
        //создаем экземпляр CounterBloc
        create: (_) => CounterBloc(),
        //передаем его в CounterPage
        child: CounterPage(),
      ),
    );
  }
}

class CounterPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Counter')),
      //CounterPage обернут в BlocBuilder
      body: BlocBuilder<CounterBloc, int>(
        // во вором параметре builder есть count === состоянию bloc
        builder: (context, count) {
          return Center(
            child: Text(
              '$count',
              style: TextStyle(fontSize: 24.0),
            ),
          );
        },
      ),
      floatingActionButton: Column(
        children: <Widget>[
          Padding(
            child: FloatingActionButton(
              child: Icon(Icons.add),
              // в context read<CounterBloc> (читаем стрим), добавляем  CounterIncrementPressed()
              onPressed: () => context.read<CounterBloc>().add(CounterIncrementPressed()),
            ),
          ),
          Padding(
            child: FloatingActionButton(
              child: Icon(Icons.remove),
              onPressed: () => context.read<CounterBloc>().add(CounterDecrementPressed()),
            ),
          ),
        ],
      ),
    );
  }
}
```

**RepositoryProvider**

Передает состояние виджетам  позволяет передать экземпляры классов

```dart
RepositoryProvider(
  create: (context) => RepositoryA(),
  child: ChildA(),
);

// with extensions
context.read<RepositoryA>();

// without extensions
RepositoryProvider.of<RepositoryA>(context)

// MultiRepositoryProvider
MultiRepositoryProvider(
  providers: [
    RepositoryProvider<RepositoryA>(
      create: (context) => RepositoryA(),
    ),
    RepositoryProvider<RepositoryB>(
      create: (context) => RepositoryB(),
    ),
    RepositoryProvider<RepositoryC>(
      create: (context) => RepositoryC(),
    ),
  ],
  child: ChildA(),
);
```
**Применение BlocProvider**

```dart
class WeatherRepository {
  WeatherRepository({WeatherApiClient? weatherApiClient})
      : _weatherApiClient = weatherApiClient ?? WeatherApiClient();

  final WeatherApiClient _weatherApiClient;

  Future<Weather> getWeather(String city) async {
    final location = await _weatherApiClient.locationSearch(city);
    final woeid = location.woeid;
    final weather = await _weatherApiClient.getWeather(woeid);
    return Weather(
      temperature: weather.theTemp,
      location: location.title,
      condition: weather.weatherStateAbbr.toCondition,
    );
  }
}


class WeatherApp extends StatelessWidget {
  //передаем класс api
  const WeatherApp({Key? key, required WeatherRepository weatherRepository})
      : _weatherRepository = weatherRepository,
        super(key: key);

  final WeatherRepository _weatherRepository;

  @override
  Widget build(BuildContext context) {
    return RepositoryProvider.value(
      value: _weatherRepository,
      child: BlocProvider(
        create: (_) => ThemeCubit(),
        child: WeatherAppView(),
      ),
    );
  }
}

class WeatherPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => WeatherCubit(context.read<WeatherRepository>()),
      child: WeatherView(),
    );
  }
}
```


```dart

```


<!-- Анимации --------------------------------------------------------------------------------------->

## Виджеты анимаций ##

### AnimatedAlign ###

```dart
AnimatedAlign(
  //направление анимации selected - состояние
  alignment: selected ? Alignment.bottomRight : Alignment.bottomLeft
  duration: const Duration(seconds: 1),curve: Curves.fastOutSlowIn,
  child: const FlutterLogo(size: 50.0),
  //необязательные свойства
  heightFactor: 1.0
  widthFactor: 1.0
  curve: Curve.fastOutSlowIn
  onEnd: VoidCallback
),

```

### AnimatedCrossFade ###

Анимация перехода между двумя элементами

```dart 
AnimatedCrossFade(
  duration: const Duration(seconds: 3),//время перехода
  firstChild: const FlutterLogo(style: FlutterLogoStyle.horizontal, size: 100.0),//первый элемент на отображение
  secondChild: const FlutterLogo(style: FlutterLogoStyle.stacked, size: 100.0),//второй 
  crossFadeState: _first ? CrossFadeState.showFirst : CrossFadeState.showSecond,//_first - флаг состояния, который определяет какой из виджетов отображать
)

```

### AnimatedDefaultTextStyle ### 

```dart 
AnimatedDefaultTextStyle(
  child, //текстовой виджет для преобразования 
  style, //целевой стиль
  duration
  )

```

### AnimatedList ### 

Позволяет анимировать действия со списком

