# поддержка html тегов метаданных

```js
function BlogPost({ post }) {
  return (
    <article>
      <h1>{post.title}</h1>
      <title>{post.title}</title>
      <meta name="author" content="Josh" />
      <link rel="author" href="https://twitter.com/joshcstory/" />
      <meta name="keywords" content={post.keywords} />
      <p>Eee equals em-see-squared...</p>
    </article>
  );
}
```

# поддержка async scripts

```js
function MyComponent() {
  return (
    <div>
      <script async={true} src="..." />
      Hello World
    </div>
  );
}

function App() {
  return (
    <html>
      <body>
        <MyComponent />
        <MyComponent /> // не приведет к дублированию скрипта в DOM
      </body>
    </html>
  );
}
```

# svg

```ts
//для инициализации в TS
declare module '*.svg' {
  import React from 'react';
  const SVG: React.VFC<React.SVGProps<SVGElement>>;
  export default SVG;
}

//в WP конфиге:
module: {
  rules: [
    {
      test: /\.svg$/,
      use: ['@svgr/webpack']
    }
  ]
},

```
