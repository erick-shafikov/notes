# transition-group

позволяет сгруппировать несколько элементов с одной анимацией

```vue
<template>
  <transition-group name="fade">
    <div class="form-group" v-for="(guest, i) in guests" :key="guest.id">
      <label @dblclick="removeGuest(i)">Guest {{ i + 1 }}</label>
      <input v-model.trim="guest.value" type="text" class="form-control" />
    </div>
  </transition-group>
</template>
<script></script>
```
