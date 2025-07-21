# TouchableWithoutFeedback

- поддерживает только одного ребенка

```js
function MyComponent(props: MyComponentProps) {
  return (
    <View {...props} _style={{ flex: 1, backgroundColor: "#fff" }}>
      <Text>My Component</Text>
    </View>
  );
}

<TouchableWithoutFeedback onPress={() => alert("Pressed!")}>
  <MyComponent />
</TouchableWithoutFeedback>;
```
