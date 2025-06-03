# Основные понятия

- dart – oop язык, статичная типизация, преобразуется в JS – код
- Использует DART Virtual machine
- При запуске программы компилятор ищет функцию main

```dart
void main(){...}
```

# Типизация

- типы: int, String, double, dynamic
- Каждое значение имеет тип и его нельзя поменять
- Типы могут вычисляться, например при возврате значения из функции

# Объявление переменных

```dart
String? x; // x == null по умолчанию
```

# операторы

**Cascade notation**

```dart
var paint = Paint() // var paint = Paint();
  ..color = Colors.black // paint.color = Colors.black
  ..strokeCap = StrokeCap.round // paint.strokeCap = StrokeCap.round;
  ..strokeWidth = 5.0; // paint.strokeWidth = 5.0;
```

**null-shorting cascade**

```dart
querySelector('#confirm') // var button = querySelector('#confirm');
  ?..text = 'Confirm' // button?.text = 'Confirm';
  ..classes.add('important') //button?.classes.add('important');
  ..onClick.listen((e) => window.alert('Confirmed!')) //button?.onClick.listen((e) => window.alert('Confirmed!'));
  ..scrollIntoView(); //button?.scrollIntoView();
```

# metadata

@Deprecated, @deprecated, @override, @pragma, @Todo

# Строки

```dart
//Интерполяция строки:
print('My name is $name');
//Интерполяция динамической строки:
'My name is ${name.length}'
```

# Функции

Если не определить возращаемое значение – возвращаемое значение будет dynamic

```dart
String myName(){ //типизация возврата обязательна
  return 'name';
}
```

**Опциональные параметры**

```dart
int sumUpToFive(int a, [int? b, int? c, int? d, int? e]) {
  int sum = a;
  if (b != null) sum += b;
  if (c != null) sum += c;
  if (d != null) sum += d;
  if (e != null) sum += e;
  return sum;
}
Значения по умолчанию
int sumUpToFive(int a, [int b = 2, int c = 3, int d = 4, int e = 5]) {
// ···
}
```

**Опциональный вызов**

```dart
//js - подобный синтаксис
var button = querySelector('#confirm');
button?.text = 'Confirm';
button?.classes.add('important');
button?.onClick.listen((e) => window.alert('Confirmed!'));
button?.scrollIntoView();

//с помощью каскада
querySelector('#confirm')
  ?..text = 'Confirm'
  ..classes.add('important')
  ..onClick.listen((e) => window.alert('Confirmed!'))
  ..scrollIntoView();
```

# Maps (объекты)

```dart
final aMapOfStringsToInts = {
  'one': 1,
  'two': 2,
  'three': 3,
};
//Типизация
final aMapOfIntToDouble = <int, double>{};

```

# Lists (массивы)

- Удаление, вставка работает по ссылке

```dart
list.add(item)// Добавить item в list
```

# Sets (множества)

```dart
//Объявление
final aSetOfStrings = {'one', 'two', 'three'};
//Типизация:
final aSetOfInts = <int>{}

```

# Перебираемые коллекции

```dart
Iterable<int> iterable = [1, 2, 3];
int value = iterable[1];//ошибка
int value = iterable.elementAt(1); //правильный доступ к элементу перебираемого объекта

void func = () => {//перебор
  for (final element in iterable) {
    print(element)
    }
};

```

# Классы

сокращение для создания экземпляра

```dart
 var person = Person('FirstName'); //var person1 = new Person('FirstName'); new - необязательно
```

## конструктор

Перенаправляющий конструктор

```dart
class Automobile {
  String make;
  String model;
  int mpg;

Automobile(this.make, this.model, this.mpg);
Automobile.hybrid(String make, String model) : this(make, model, 60);//перенаправит в главный конструктор 60
Automobile.fancyHybrid() : this.hybrid('FutureCar', 'Mark 2'); //перенаправит в hybrid
}
```

Объект в качестве конструктора

```dart
class Person {
  late String firstName;
  Person(this.firstName);//Person(name){firstname = name}
  Person({required this.red, required this.green, required this.blue}); //в качестве параметра - объект

  printName(){//print(this.firstName); - this - необязательно
    print(firstName);
  }
}
```

именованный конструктор

```dart
class ImageModel {
  late int id;
  late String url;
  late String title;
ImageModel(this.id, this.url, this.title);
//если нужно предопределить что-то перед инициализацией параметров в конструкторе
ImageModel.fromJson(Map<String, dynamic> parsedJson) {
    id = parsedJson['id'];
    url = parsedJson['url'];
    title = parsedJson['title'];
  }
}

```

Константный конструктор

```dart
class ImmutablePoint {
  static const ImmutablePoint origin = ImmutablePoint(0, 0);
  final int x;
  final int y;
const ImmutablePoint(this.x, this.y);
}
```

# Исключения

```dart
try {
  breedMoreLlamas();
} on OutOfLlamasException {// Проверка на особое исключение
    buyMoreLlamas();
} on Exception catch (e) {// Поймает все исключения
    print('Unknown exception: $e');
} catch (e) {
    print('Something really unknown: $e'); //Поймает все
} finally {
    cleanLlamaStalls(); // сработает в любом случае
}

```

Проброс

```dart
try {…} catch (e) {rethrow;} //Проброс дальше
```

<!--- Streams --------------------------------------------------------------------------------------------->

# Stream

Поток асинхронных данных - повторяющиеся Future
Два типа потоков - стрим с single subscriptions и broadcasting

```dart
//каждый stream вернет число
Future<int> sumStream(Stream<int> stream) async {
  var sum = 0;
  await for (final value in stream) {
    sum += value;

  }
  return sum;
}
```

**lastWhere** позволяет найти по условию

```dart
Future<int> lastPositive(Stream<int> stream) =>
    stream.lastWhere((x) => x >= 0);
```

```dart
var counterStream =
    Stream<int>.periodic(const Duration(seconds: 1), (x) => x).take(15);

counterStream.forEach(print); // Print an integer every second, 15 times.

```

```dart

```
