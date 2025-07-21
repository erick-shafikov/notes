# RefreshControl

используется для жеста вниз на самом верху (обновление)

```js
<RefreshControl
  colors={String[]} //(Android) цвет индикатора
  enabled={true} //(Android) состояние обновления
  onRefresh={Function} // коллбек на срабатывание
  progressBackgroundColor={type Color} //(Android)
  progressViewOffset={Number} //отступ от верха
  refreshing={Boolean!} // индикатор
  size={'default', 'large'}//(Android)
  tintColor={type Color}//(IOS)
  title={String}//(IOS)
  titleColor={type Color}//(IOS)
/>
```

```js
const RefreshControl = () => {
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  return (
    <SafeAreaView>
      <ScrollView
        //  ScrollView принимает проп refreshControl
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <Text>Pull down to see RefreshControl indicator</Text>
      </ScrollView>
    </SafeAreaView>
  );
};
```
