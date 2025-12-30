# teleport

Позволяет пробросить в конкретное место

```vue
<Teleport to="#some-id" />
<Teleport to=".some-class" />
<Teleport to="[data-teleport]" />
```

```vue
<Teleport to="#popup" :disabled="displayVideoInline">
  <video src="./my-movie.mp4">
</Teleport>
```
