# стилизация

## css-модули

Поддерживается модульная стилизация из коробки, стили модифицируются и HMR доступен по умолчанию. Использование глобальных стилей предусматривается импортом в файл app

```js
import styles from "./styles.module.css";
export default function RootLayout(){
 …
  return (
   <div className={styles.container}>
    );
}

```
