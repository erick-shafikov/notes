# AnimationEvent

интерфейс, который содержит информацию об анимации

Наследование - Event

конструктор:

new AnimationEvent(type, options)

- type - animationstart, animationend, or animationiteration
- options:
- - animationName
- - elapsedTime
- - pseudoElement

свойства:

- animationName - имя анимации
- elapsedTime - количество секунд с начала анимации
- pseudoElement - имя псевдоэлемента

методы:

- initAnimationEvent() - инициализирует AnimationEvent
