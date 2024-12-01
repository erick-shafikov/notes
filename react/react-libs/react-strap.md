<!-- Modal ------------------------------------------------------------------------------------------------------------------------------------------------------------------>

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

<!-- Card ------------------------------------------------------------------------------------------------------------------------------------------------------------------>

# Card

Компонент карточки всегда занимает 100% контейнера

```jsx
import {
  Card,
  Button,
  CardBody,
  CardFooter,
  CardLink,
  CardSubtitle,
  CardText,
  CardTitle,
  Col,
  ListGroup,
  ListGroupItem,
  CardImg,
} from "react-strap";

// ----------------------------------------------------------------------

<Card
  _style={{
    width: "18rem",
  }}
  color="primary" //цвет карточки
  outline //рамка будет работать только в паре с color
  inverse // цвет текста
  body //добавить отступов от контента
  // У всех карточек ----------------------------------------------------
  className="text-center" //выравнивание всего текст внутри карточки
  tag="div"
  innerRef={ref}
  cssModule
>
  {/*комопнент заголовок отделенный в такой же контейнер как и footer*/}
  <CardHeader>Header</CardHeader>
  {/*комопнент заголовок-картинка*/}
  <CardImg
    alt="Card image cap"
    src="https://picsum.photos/900/180"
    _style={{
      height: 180,
    }}
    top
    width="100%"
    bottom //если внизу закруглит края у картинки
    top //если сверху закруглит края у картинки
  />
  <img alt="Sample" src="https://picsum.photos/300/200" />
  {/*тело карточки таких компонентов может быть несколько*/}
  <CardBody>
    <CardTitle tag="h5">Card title</CardTitle>
    <CardSubtitle className="mb-2 text-muted" tag="h6">
      Card subtitle
    </CardSubtitle>
    <CardText>
      Some quick example text to build on the card title and make up the bulk of
      the card‘s content.
    </CardText>
    <Button>Button</Button>
  </CardBody>
  {/*низ карточки*/}
  <CardFooter>Footer</CardFooter>
  {/*комопнент подвал-картинка*/}
  <CardImg
    alt="Card image cap"
    src="https://picsum.photos/900/180"
    style={{
      height: 180,
    }}
    top
    width="100%"
  />
</Card>;
```

Пример карточки с картинкой позади

```jsx
import { Card, CardText, CardTitle, CardImg } from "react-strap";

<Card inverse>
  <CardImg
    alt="Card image cap"
    src="https://picsum.photos/900/270?grayscale"
    style={{
      height: 270,
    }}
    width="100%"
  />
  {/*в паре с CardImg обернут задний фон в картинку*/}
  <CardImgOverlay>
    <CardTitle tag="h5">...</CardTitle>
    <CardText>...</CardText>
  </CardImgOverlay>
</Card>;
```

## CardGroup - горизонтальные группы карточек

Объединяет карточки в группу

```jsx
<CardGroup>
  <Card>
    <CardBody>...</CardBody>
  </Card>
  <Card>
    <CardBody>...</CardBody>
  </Card>
  <Card>
    <CardBody>...</CardBody>
  </Card>
</CardGroup>
```

## CardColumns - вертикальные группы карточек

```jsx
<CardColumns
  _style={{
    width: "18rem",
  }}
>
  <Card>
    <CardBody>...</CardBody>
  </Card>
  <Card>
    <CardBody>...</CardBody>
  </Card>
  <Card>
    <CardBody>...</CardBody>
  </Card>
</CardColumns>
```
