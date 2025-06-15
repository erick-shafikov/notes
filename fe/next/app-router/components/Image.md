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

может быть ссылка на локальный файл или не удаленный, если удаленный, то нужно настроить. Может принимать экспортированный файл

- самостоятельно определит размер

```tsx
import someImageSrc from "@/assets/some-image.png";

<Image src={someImageSrc} />;
```

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

путь до данных если при placeholder "blur", Это должно быть 64-base. Например изображение сжатое до 10X10 px

## unoptimized

```js
unoptimized = {false} // {false} | {true}
```

## overrideSrc

для подмены изображения

## decoding

async, sync, auto

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

# настройки config

```js
module.exports = {
  images: {
    //для блокировки других путей
    localPatterns: [
      {
        pathname: "/assets/images/**",
        search: "",
      },
    ],
    //для удаленных картинок
    remotePatterns: [
      {
        protocol: "https",
        hostname: "example.com",
        port: "",
        pathname: "/account123/**",
        search: "?v=1727111025337",
      },
    ],
    //лоудеры при отключении автомтической оптимизации
    loader: "custom",
    loaderFile: "./my/image/loader.js",
    // если известно на каких устройствах
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    //размеры картинок
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    //допустимые значения качества
    qualities: [25, 50, 75],
    formats: ["image/webp"],
    //конфигурация кеша
    minimumCacheTTL: 60, // 1 minute
    //запретить статический импорт
    disableStaticImages: true,
  },
};
```

# BPs

## Отзывчивые изображения

```tsx
import Image from "next/image";
import me from "../photos/me.jpg";

export default function Author() {
  return (
    <>
      {/* со статическим импортом */}
      <Image
        src={me}
        alt="Picture of the author"
        sizes="100vw"
        style={{
          width: "100%",
          height: "auto",
        }}
      />
      {/* со динамическим импортом */}
      <Image
        src={photoUrl}
        alt="Picture of the author"
        sizes="100vw"
        style={{
          width: "100%",
          height: "auto",
        }}
        width={500}
        height={300}
      />
      {/* fill */}
      <div style={{ position: "relative", width: "300px", height: "500px" }}>
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
    </>
  );
}
```

## art direction

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

## background images

```tsx
import { getImageProps } from "next/image";

function getBackgroundImage(srcSet = "") {
  const imageSet = srcSet
    .split(", ")
    .map((str) => {
      const [url, dpi] = str.split(" ");
      return `url("${url}") ${dpi}`;
    })
    .join(", ");
  return `image-set(${imageSet})`;
}

export default function Home() {
  const {
    props: { srcSet },
  } = getImageProps({ alt: "", width: 128, height: 128, src: "/img.png" });
  const backgroundImage = getBackgroundImage(srcSet);
  const style = { height: "100vh", width: "100vw", backgroundImage };

  return (
    <main _style={style}>
      <h1>Hello World</h1>
    </main>
  );
}
```
