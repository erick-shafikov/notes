# Пирамида тестирования

- **E2E** – на реальных данных, тест сам проверяет функциональность важных частей
- **Интеграционные** – тест модулей (unit) в связке
- **unit** – тестирования, отдельно единицы (функции, классы)
- **скриншот тесты**

Принципы:

- проверяем на часто встречающихся данных
- проверяем на крайние значения
- Проверяем ошибки

# JEST

npm run test – проход по всем тестам
npm run test validateValue.test.js – тест отдельной единицы