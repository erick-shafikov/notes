# Slots

```vue
<!-- компонент со слотом -->
<template>
  <button class="button">
    <div>
      <slot name="icon"></slot>
    </div>
    <slot></slot>
  </button>
</template>

<!-- использование -->
<template>
  <!-- поддерживает динамические #[id] и статичные слоты -->
  <Button>
    <!-- шк v-slot='icon' -->
    <template #icon>+</template>
    Сохранить</Button
  >
</template>
```

# именованный слот со значением по умолчанию

```vue
<template>
  <div class="alert" :class="classes">
    <h4>{{ title }}</h4>
    <hr />
    <div>
      <slot></slot>
    </div>
    <hr />
    <slot name="footer">
      <!-- значение по умолчанию -->
      <button type="button" class="btn btn-primary" @click="$emit('ok')">
        Ok
      </button>
    </slot>
  </div>
</template>
```

```vue
<template>
  <app-alert title="Hello, user">
    <!-- анонимный слот -->
    <p>Total: {{ total }}</p>
    <p>Items: {{ items }}</p>
    <!-- именованный слот с компонентом по умолчанию -->
    <template v-slot:footer>
      <button type="button" class="btn btn-danger mr-1" @click="reset">
        Reset
      </button>
      <button type="button" class="btn btn-primary">Send</button>
    </template>
  </app-alert>
</template>
```

# v-if + slots

```vue
<template>
  <div class="card">
    <div v-if="$slots.header" class="card-header">
      <slot name="header" />
    </div>

    <div v-if="$slots.default" class="card-content">
      <slot />
    </div>

    <div v-if="$slots.footer" class="card-footer">
      <slot name="footer" />
    </div>
  </div>
</template>
```

# scoped slots

```vue
<template>
  <!-- <MyComponent> -->
  <div>
    <slot :text="greetingMessage" :count="1"></slot>
  </div>
</template>
```

```vue
<template>
  <MyComponent v-slot="slotProps">
    {{ slotProps.text }} {{ slotProps.count }}
  </MyComponent>
</template>
```
