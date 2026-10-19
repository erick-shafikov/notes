```tsx
ReactDOM.createPortal(child, container);
```

РЁР°Рі 1. Р”РѕР±Р°РІСЊС‚Рµ РєРѕСЂРЅРµРІРѕР№ СЌР»РµРјРµРЅС‚ РІ HTML

```html
<div id="modal-root"></div>
```

```tsx
// РЁР°Рі 2. РЎРѕР·РґР°Р№С‚Рµ РєРѕРјРїРѕРЅРµРЅС‚ Modal:
import React from "react";
import ReactDOM from "react-dom";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal = ({ isOpen, onClose, children }: ModalProps): JSX.Element | null => {
  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          Г—
        </button>
        {children}
      </div>
    </div>,
    document.getElementById("modal-root")
  );
};

export default Modal;
```

РџСЂРёРјРµСЂ РІ РїСЂРёР»РѕР¶РµРЅРёРё:

```tsx
import React, { useState } from "react";
import Modal from "./Modal";

function App(): JSX.Element {
  const [isModalOpen, setModalOpen] = useState<boolean>(false);

  return (
    <div>
      <button onClick={() => setModalOpen(true)}>РћС‚РєСЂС‹С‚СЊ РјРѕРґР°Р»СЊРЅРѕРµ РѕРєРЅРѕ</button>
      <Modal isOpen={isModalOpen} onClose={() => setModalOpen(false)}>
        <h1>Р±РѕР»СЊС€РѕР№ РїСЂРёРІРµС‚ РѕС‚ ReactJs Daily!</h1>
        <p>РџРѕРґРїРёС€РёСЃСЊ РЅР° РєР°РЅР°Р»)</p>
      </Modal>
    </div>
  );
}

export default App;
```
