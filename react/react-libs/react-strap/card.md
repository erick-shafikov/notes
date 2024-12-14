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

# CardGroup - горизонтальные группы карточек

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

# CardColumns - вертикальные группы карточек

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

# CSS

#Card

```html
<div class="card">
  <!-- content -->
</div>
```

```scss
[dir] .card {
  margin-bottom: 2rem;
  box-shadow: 0 4px 24px 0 rgba(34, 41, 47, 0.1);
}

[dir] .card {
  background-color: #fff;
  background-clip: border-box;
  border: 0 solid rgba(34, 41, 47, 0.125);
  border-radius: 0.428rem;
}

.card {
  transition: all 0.3s ease-in-out, background 0s, color 0s, border-color 0s;
}

.card {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  word-wrap: break-word;
}
```
