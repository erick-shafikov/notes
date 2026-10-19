# РїРѕРґРґРµСЂР¶РєР° html С‚РµРіРѕРІ РјРµС‚Р°РґР°РЅРЅС‹С…

```tsx
interface BlogPostProps {
  post: { title: string; keywords: string };
}

function BlogPost({ post }: BlogPostProps): JSX.Element {
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

# РїРѕРґРґРµСЂР¶РєР° async scripts

```tsx
function MyComponent(): JSX.Element {
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
        <MyComponent /> // РЅРµ РїСЂРёРІРµРґРµС‚ Рє РґСѓР±Р»РёСЂРѕРІР°РЅРёСЋ СЃРєСЂРёРїС‚Р° РІ DOM
      </body>
    </html>
  );
}
```

# svg

```ts
//РґР»СЏ РёРЅРёС†РёР°Р»РёР·Р°С†РёРё РІ TS
declare module '*.svg' {
  import React from 'react';
  const SVG: React.VFC<React.SVGProps<SVGElement>>;
  export default SVG;
}

//РІ WP РєРѕРЅС„РёРіРµ:
module: {
  rules: [
    {
      test: /\.svg$/,
В  В  В  use: ['@svgr/webpack']
В  В  }
  ]
},

```
