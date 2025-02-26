# Image

# props:

```tsx
import Image from "next/image";

export default function Page() {
  // обязательные пропсы
  return (
    <div>
      <Image
        src="/profile.png"
        {/* или fill */}
        width={500}
        height={500}
        alt="Picture of the author"
      />
    </div>
  );
}
```

## src

может быть ссылка на локальный файл или не удаленный, если удаленный, то нужно настроить

### загрузка сторонних

Для удаленных нужно добавлять в next.config

```js
//next.config
images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "miro.medium.com",
        port: "",
        pathname: "/**",
      },
    ],
  },

```

localPatterns - для локальных
remotePatterns - для удаленных
domains - массив с допустимыми доменами
loaderFile -
deviceSizes - массив с размерами
imageSizes -
qualities - значения качеств
formats - форматы
minimumCacheTTL - настройка кеширования изображений
disableStaticImages - отключить статические файлы
dangerouslyAllowSVG - разрешить использовать в качестве src svg файлы
contentDispositionType

## width hright

размер в пикселях

## alt

может быть пустой строкой

## loader

функция для обработки url, можно настроить в next.config.js

```tsx
// loader пример
import Image from "next/image";

const imageLoader = ({ src, width, quality }) => {
  return `https://example.com/${src}?w=${width}&q=${quality || 75}`;
};

export default function Page() {
  return (
    <Image
      loader={imageLoader}
      src="me.png"
      alt="Picture of the author"
      width={500}
      height={500}
    />
  );
}
```

## fill

- если hright и width неизвестны.
- Родитель должен быть position равный relative, fixed или absolute, так как у img position === absolute.
- object-fit === contain что бы вписать в контейнер, cover для обрезки но сохранения соотношения сторон

Отзывчивое изображение с помощью fill

```tsx
import Image from "next/image";

export default function Page({ photoUrl }) {
  return (
    <div _style={{ position: "relative", width: "300px", height: "500px" }}>
      <Image
        src={photoUrl}
        alt="Picture of the author"
        sizes="300px"
        fill
        style={{
          objectFit: "contain",
        }}
      />
    </div>
  );
}
```

## size:

srcset строка для разных viewport

```tsx
//пример
export default function Page() {
  return (
    <div className="grid-element">
      <Image
        fill
        src="/example.png"
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
      />
    </div>
  );
}

import Image from "next/image";
import me from "../photos/me.jpg";

export default function Author() {
  return (
    <Image
      src={me}
      alt="Picture of the author"
      sizes="100vw"
      style={{
        width: "100%",
        height: "auto",
      }}
    />
  );
}
```

## quality

значение от 1 до 100, по умолчанию 75

## priority

Если true до будет предварительно загружен

## placeholder

```ts
placeholder = "empty"; // "empty" | "blur" | "data:image/..."
```

если blur то blurDataUrl будет использовано как заполнитель, нужно для удаленных изображений, в случае "data:image/..." - адрес данных

## style

для css стилей

## onLoadingComplete, onLoad, onError

```tsx
const Img = () => {
  const callBack = (imageOrError) => console.log(imageOrError);
  return (
    <>
      {/* изображение загрузилось */}
      <Image onLoadingComplete={callBack} />
      {/* после удаления заполнителя */}
      <Image onLoad={callBack} />
      {/* ошибка */}
      <Image onError={callBack} />
    </>
  );
};
```

## loading

```js
loading = "lazy"; // {lazy} | {eager}
```

## blurDataURL

путь до данных если при placeholder "blur"

## unoptimized

```js
unoptimized = {false} // {false} | {true}
```

## overrideSrc

для подмены изображения

<!-- getImageProps() ------------------------------------------------------------------------>

# getImageProps()

Реализация загрузки разных изображения в зависимости от темы

```tsx
import { getImageProps } from "next/image";

export default function Page() {
  const common = { alt: "Theme Example", width: 800, height: 400 };

  const {
    props: { srcSet: dark },
  } = getImageProps({ ...common, src: "/dark.png" });

  const {
    props: { srcSet: light, ...rest },
  } = getImageProps({ ...common, src: "/light.png" });

  return (
    <picture>
      <source media="(prefers-color-scheme: dark)" srcSet={dark} />
      <source media="(prefers-color-scheme: light)" srcSet={light} />
      <img {...rest} />
    </picture>
  );
}
```

альтернатива через css

```css
.imgDark {
  display: none;
}

@media (prefers-color-scheme: dark) {
  .imgLight {
    display: none;
  }
  .imgDark {
    display: unset;
  }
}
```

```tsx
import styles from "./theme-image.module.css";
import Image, { ImageProps } from "next/image";

type Props = Omit<ImageProps, "src" | "priority" | "loading"> & {
  srcLight: string;
  srcDark: string;
};

const ThemeImage = (props: Props) => {
  const { srcLight, srcDark, ...rest } = props;

  return (
    <>
      <Image {...rest} src={srcLight} className={styles.imgLight} />
      <Image {...rest} src={srcDark} className={styles.imgDark} />
    </>
  );
};
```

для разных устройств разные изображения

```tsx
import { getImageProps } from "next/image";

export default function Home() {
  const common = { alt: "Art Direction Example", sizes: "100vw" };

  const {
    props: { srcSet: desktop },
  } = getImageProps({
    ...common,
    width: 1440,
    height: 875,
    quality: 80,
    src: "/desktop.jpg",
  });

  const {
    props: { srcSet: mobile, ...rest },
  } = getImageProps({
    ...common,
    width: 750,
    height: 1334,
    quality: 70,
    src: "/mobile.jpg",
  });

  return (
    <picture>
      <source media="(min-width: 1000px)" srcSet={desktop} />
      <source media="(min-width: 500px)" srcSet={mobile} />
      <img {...rest} style={{ width: "100%", height: "auto" }} />
    </picture>
  );
}
```
