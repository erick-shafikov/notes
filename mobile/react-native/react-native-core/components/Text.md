# Text

- в него может быть вложен только другой <Text />
- поддерживает интерактивность
- Inline элементы
- рекомендуется создать текстовый компонент для всего

```js
<Text
//Accessibility
accessibilityHint={String}
accessibilityLanguage={String}
accessibilityLabel={String}
accessibilityRole={type AccessibilityRole}
accessibilityState={type AccessibilityState}
accessibilityState={Array}
onAccessibilityAction={Function}
accessible={true} //активация Accessibility
adjustsFontSizeToFit={false} //автоматическое уменьшение стиля
allowFontScaling={true} //масштабирование текста
ellipsizeMode={'head'| 'middle'| 'tail'| 'clip'} //работает в паре с numberOfLines определяет как декорировать окончание строки
id={String}
maxFontSizeMultiplier={Number} // если allowFontScaling  === true определяет максимальное увеличение текста
minimumFontScale={Number} //
nativeID={String}
numberOfLines={0}
onLayout={({nativeEvent: LayoutEvent}) => void} //вызывается на монтирование и изменение
onLongPress={({nativeEvent: PressEvent}) => void}
onMoveShouldSetResponder={({nativeEvent: PressEvent}) => boolean}
onPress={({nativeEvent: PressEvent}) => void} //  onPress = onPressIn + onPressOut
onPressIn={({nativeEvent: PressEvent}) => void}
onPressOut={({nativeEvent: PressEvent}) => void}
//обработка движений
onResponderGrant={({nativeEvent: PressEvent}) => void ｜ boolean}
onResponderMove={({nativeEvent: PressEvent}) => void}
onResponderRelease={({nativeEvent: PressEvent}) => void}
onResponderTerminate={({nativeEvent: PressEvent}) => void}
onResponderTerminationRequest={({nativeEvent: PressEvent}) => boolean}
onStartShouldSetResponderCapture={({nativeEvent: PressEvent}) => boolean}

onTextLayout={(TextLayoutEvent) => mixed}
pressRetentionOffset={type Rect | Number}
role={type Role} //
selectable={false}//позволяет выделить текст, по умолчанию текст выделять нельзя
_style={} //
testID={} //
//aria props
aria-busy={false}
aria-disabled={false}
aria-expanded={false}
aria-label={String}
aria-selected={Boolean}
//Android only props
android_hyphenationFrequency={'none' | 'normal' | 'full'} //расстановка переносов
dataDetectorType={'none' | 'phoneNumber' | 'link' | 'email' | 'all'} //превратить в интерактивный текст
disabled={false} //
selectionColor ={type Color} //цвет выделения текста
textBreakStrategy ={'simple'|'highQuality'|'balanced'}
//iOS props only
dynamicTypeRamp={'caption2'|'caption1'|'footnote'|'subheadline'|'callout'|'body'|'headline'|'title3'|'title2'|'title1'|'largeTitle'}
/>
suppressHighlighting={false} //при нажатии на  текст
lineBreakStrategyIOS={'none'|'standard'|'hangul-word'|'push-out'}
```
