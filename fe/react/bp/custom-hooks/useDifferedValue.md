## useDifferedValue

```tsx
const useDifferedValue = <T extends string | number>(value: T) => {
  const [returnValue, setReturnValue] = useState(value);
  const [ignore, setIgnore] = useState(true);

  useEffect(() => {
    const id = setTimeout(() => {
      setIgnore(false);
    }, 2000);
    return () => clearInterval(id);
  }, [value]);

  useEffect(() => {
    if (!ignore) {
      setReturnValue(value);
    }
    return () => {
      setIgnore(true);
    };
  }, [ignore]);

  return returnValue;
};
```
