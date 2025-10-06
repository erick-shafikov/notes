# MantineProvider

Провайдер контекста для компонентов

Пропсы:

# theme

[объект темы](../objects/theme.md)

# colorSchemeManager

утилита для управления логикой цветовой схемой [пример реализации colorSchemeManager](../functions/localStorageColorSchemeManager.md)

# cssVariablesSelector

куда будет добавлены переменные, по умолчанию :root

# withCssVariables

true, позволяет:

- Создаёт глобальные :root переменные;
- Следит за изменениями темы (например, при смене dark/light);
- Автоматически обновляет переменные.

# deduplicateCssVariables

true - будут ли удалены переменные дублирующие тему

# getRootElement

в какой элемент html вставить mantine-color-scheme

# classNamesPrefix

поменять префикс mantine на кастомный

# withStaticClasses

true, включить ли статические классы

# withGlobalClasses

будет ли добавлены глобальные классы в style тег

# getStyleNonce

для генерации nonce атрибута в стиль тег

# env

'default' - окружение, возможные значения: default, test
