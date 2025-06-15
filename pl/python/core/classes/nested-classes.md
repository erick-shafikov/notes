```python
class Women:
    title = 'title'
    photo = 'photo'
    ordering = 'ordering of Women'

    class Meta:
        ordering = ['id']

        def __init__(self, access):
            self._access = access

    def __init__(self, user, psw):
        self.user = user
        self.psw = psw
        # что бы создать экземпляр Meta c Women
        self.meta = self.Meta(123)
        # так можно, наоборот нельзя - обращаться к классу Women из Meta


w = Women('x', 123)

print(w.ordering)
print(w.Meta.ordering)
```