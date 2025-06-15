flashSync позволяет немедленно обновить DOM

```jsx
import { useState, useEffect } from "react";
import { flushSync } from "react-dom";

export default function PrintApp() {
  const [isPrinting, setIsPrinting] = useState(false);

  useEffect(() => {
    // вызовется перед печатью и обновит состояние
    // если этого не сделать, то при открытии диалогового окну печати isPrinting === false Это происходит потому, что React пакетирует обновления асинхронно, и диалоговое окно печати отображается до обновления состояния.
    function handleBeforePrint() {
      flushSync(() => {
        setIsPrinting(true);
      });
    }

    function handleAfterPrint() {
      setIsPrinting(false);
    }

    window.addEventListener("beforeprint", handleBeforePrint);
    window.addEventListener("afterprint", handleAfterPrint);
    return () => {
      window.removeEventListener("beforeprint", handleBeforePrint);
      window.removeEventListener("afterprint", handleAfterPrint);
    };
  }, []);

  return (
    <>
      <h1>isPrinting: {isPrinting ? "yes" : "no"}</h1>
      <button onClick={() => window.print()}>Print</button>
    </>
  );
}
```
