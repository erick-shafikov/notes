Вложенность реализуется посредству вложения одного окна в другое, то есть дочернее модальное окно должно быть дочерним компонентом

# Modal

```jsx
import { useState } from "react";
import { Button, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap";

function Example() {
  const [modal, setModal] = useState(false);

  const toggle = () => setModal(!modal);
  //пользовательская кнопка для заголовка
  const closeBtn = (
    <button className="close" onClick={toggle} type="button">
      &times;
    </button>
  );
  //пользовательская кнопка для модального окна
  const externalCloseBtn = (
    <button
      type="button"
      className="close"
      _style={{ position: "absolute", top: "15px", right: "15px" }}
      onClick={toggle}
    >
      &times;
    </button>
  );

  return (
    <div>
      <Button color="danger" onClick={toggle}>
        Click Me
      </Button>
      <Modal
        autoFocus
        backdrop // задний фон
        backdropClassName="some-bs-class-name" //стили заднего фона
        backdropTransition={{ timeout: 1300 }} // появление заднего фона
        centered // центрирование
        contentClassName="some-bs-class-name" //стили контента
        className="some-bs-class-name"
        external={externalCloseBtn} // компонент закрытия кнопки
        fade // анимация появления исчезновения
        fullscreen // fullscreen для всех режимов
        fullscreen="sm | md | lg | xs" // fullscreen для адаптивных режимов
        innerRef={ref}
        isOpen={Boolean}
        keyboard // реагирование на клавиатуру
        // labelledBy=""
        modalClassName="some-bs-class-name"
        modalTransition={{ timeout: 700 }} // появление окна
        onClosed={() => {}} // коллбек на закрытие
        onEnter={() => {}} // коллбек на открытие
        onExit={() => {}} // коллбек
        onOpened={() => {}} // коллбек
        returnFocusAfterClose // установка фокуса
        role=""
        scrollable // прокрутка при переполнении
        size="sm | md | lg" // размеры
        toggle={toggle}
        tapFocus={false}
        // fullscreen для адаптивных режимов
        unmountOnClose={unmountOnClose} // сбрасывать данные при закрытие
        wrapClassName="some-bs-class-name"
        zIndex="1050"
      >
        <ModalHeader
          toggle={toggle} //для добавления кнопки закрытия
          close={closeBtn} //для добавления пользовательского компонента кнопки
          wrapTag="div"
        >
          Modal title
        </ModalHeader>
        <ModalBody></ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={toggle}>
            Do Something
          </Button>
          <Button color="secondary" onClick={toggle}>
            Cancel
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
```
