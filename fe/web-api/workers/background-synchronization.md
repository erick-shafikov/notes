Помогает откладывать задачи, что бы можно было запустить из в service worker, как только появится соединение. Работает с помощью интерфейсов SyncManager

# SyncManager

методы:

- register(tag) ⇒ промис tag - id из события синхронизации
- getTags() ⇒ лист с идентификаторами

```js
async function syncMessagesLater() {
  const registration = await navigator.serviceWorker.ready;
  try {
    await registration.sync.register("sync-messages");
  } catch {
    console.log("Background Sync could not be registered!");
  }
}
```

# SyncEvent

событие, которое возникает ServiceWorkerGlobalScope

свйоства:

- tag
- lastChance
