# concurrent mode

CCMode – это режим работы react-приложения, когда приложение остается отзывчивым и не блокируется, даже если на фона происходят большие вычисления. Аналогично git ведутся работы над разными изменениями в dom. Пример с вводом текста в поле input, при котором блокируется UI. debounce и throttling могут так же вызывать. Смысл в том, что react может работать параллельно над несколькими обновлениями состояния. Данный режим изначально назывался асинхронные компоненты, конкурентный режим, теперь concurrent features

Хуки которые использует CC-mode useTransition, startTransition, useDeferredValue

Приоритетные задачи на обновления:

- HIGH: useState, useReducer, useSyncExternalStore
- LOW: все остальное

Принципы выполнения задач:

- высокоприоритетные такси прерывают низкоприоритетные
- выполнение функции компонента всегда доходит до return, компонент нельзя прервать
- низкоприоритетные запоминают свое состояние
