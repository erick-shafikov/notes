Для создания объекта микроформатов h-_в атрибуте class используются имена классов.
Чтобы добавить свойство к объекту, имена классов p-_, u-_, dt-_, e- используются для одного из потомков объекта.

h-карта
Микроформат h-card представляет человека или организацию.

Свойство Описание

- p-name Полное/отформатированное имя человека или организации.
- u-email Адрес электронной почты
- u-photo фотография человека или организации
- u-url домашняя страница или другой URL-адрес, представляющий человека или организацию
- u-uid универсальный уникальный идентификатор, желательно канонический URL
- p-street-address номер дома + название
- p-locality город/поселок/деревня
- p-country-name название страны

```html
<div class="h-card">
  <a class="p-name u-url" href="https://blog.lizardwrangler.com/">
    Mitchell Baker
  </a>
  (<a class="p-org h-card" href="https://mozilla.org/">Mozilla Foundation</a>)
</div>
```

```json
{
  "items": [
    {
      "type": ["h-card"],
      "properties": {
        "name": ["Mitchell Baker"],
        "url": ["https://blog.lizardwrangler.com/"],
        "org": [
          {
            "value": "Mozilla Foundation",
            "type": ["h-card"],
            "properties": {
              "name": ["Mozilla Foundation"],
              "url": ["https://mozilla.org/"]
            }
          }
        ]
      }
    }
  ]
}
```

h-entry

- p-name entry name/title
- p-author who wrote the entry, optionally embedded h-card
- dt-published when the entry was published
- p-summary short entry summary
- e-content full content of the entry

h-feed

- p-name name of the feed
- p-author author of the feed, optionally embed an h-card

h-event

- p-name название события (или заголовок)
- p-summary краткий обзор мероприятия
- dt-start дата и время начала события
- dt-end дата и время окончания события
- p-location где происходит событие, опционально встроенная h-карта
