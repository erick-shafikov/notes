---
name: publish-to-git
description: >
  Публикует текущие изменения в GitLab: git add ., git commit -m 'upd', git push origin main.
  Триггеры: "опубликуй изменения", "залей на гит", "залей на git", "запушь",
  "push changes", "publish changes", "отправь в gitlab", "залей в репо".
---

## Steps

1. Run `git add .`
2. Run `git commit -m 'upd'` (если нечего коммитить — сообщить пользователю и остановиться)
3. Run `git push origin main`
4. Report result: что было закоммичено, статус пуша
