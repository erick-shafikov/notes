```js
ReactDOM.createPortal(child, container);
```

Шаг 1. Добавьте корневой элемент в HTML

```html
<div id="modal-root"></div>
```

```js
// Шаг 2. Создайте компонент Modal:
import React from "react";
import ReactDOM from "react-dom";

const Modal = ({ isOpen, onClose, children }) => {
  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          ×
        </button>
        {children}
      </div>
    </div>,
    document.getElementById("modal-root")
  );
};

export default Modal;
```

Пример в приложении:

```js
import React, { useState } from "react";
import Modal from "./Modal";

function App() {
  const [isModalOpen, setModalOpen] = useState(false);

  return (
    <div>
      <button onClick={() => setModalOpen(true)}>Открыть модальное окно</button>
      <Modal isOpen={isModalOpen} onClose={() => setModalOpen(false)}>
        <h1>большой привет от ReactJs Daily!</h1>
        <p>Подпишись на канал)</p>
      </Modal>
    </div>
  );
}

export default App;
```
