# TextInput

**пропсы**

- onChangeText - коллбек на изменение текста
- value дял контролируемого значения
- maxLength - максимальное количество символов

```js
<TextInput
onChangeText={(text:string) => void}
allowFontScaling={true} //Указывает, должны ли шрифты масштабироваться в соответствии с настройками специальных возможностей размера текста
autoCapitalize={'none' | 'sentences' | 'words' | 'characters'} //режим заглавной буквы
autoComplete={'additional-name'|'address-line1'|'address-line2'|'birthdate-day'|'birthdate-full'|'birthdate-month'|'birthdate-year'|'cc-csc'|'cc-exp'|'cc-exp-day'|'cc-exp-month'|'cc-exp-year'|'cc-number'|'country'|'current-password'|'email'|'family-name'|'given-name'|'honorific-prefix'|'honorific-suffix'|'name'|'new-password'|'off'|'one-time-code'|'postal-code'|'street-address'|'tel'|'username'|'cc-family-name'|'cc-given-name'|'cc-middle-name'|'cc-name'|'cc-type'|'nickname'|'organization'|'organization-title'|'url'|'gender'|'name-family'|'name-given'|'name-middle'|'name-middle-initial'|'name-prefix'|'name-suffix'|'password'|'password-new'|'postal-address'|'postal-address-country'|'postal-address-extended'|'postal-address-extended-postal-code'|'postal-address-locality'|'postal-address-region'|'sms-otp'|'tel-country-code'|'tel-device'|'tel-national'|'username-new'}
autoCorrect={true} // автокоррекция componentDidMount или useEffect
autoFocus={false} // автофокус при componentDidMount
blurOnSubmit={true} // потеря фокуса при окончании ввода текста
caretHidden={false} // скрытие каретки
contextMenuHidden={false} // скрытие меню
defaultValue={String}
editable={true}
enterKeyHint={'enter' | 'done' | 'next' | 'previous' | 'search' | 'send'}
inputMode={'decimal' | 'email' | 'none' | 'numeric' | 'search' | 'tel' | 'text' | 'url'}
keyboardType={'default'|'email-address'|'numeric'|'phone-pad'|'ascii-capable'|'numbers-and-punctuation'|'url'|'number-pad'|'name-phone-pad'|'decimal-pad'|'twitter'|'web-search'|'visible-password'} //тип клавиатуры под input
maxFontSizeMultiplier={0} // если allowFontScaling  === true то определяет число максимального увеличения шрифта
maxLength={0}
multiline={false}
onBlur={Function} // коллбек на потерю фокуса
onChange={({nativeEvent: {eventCount, target, text}}) => void} // коллбек на изменение
onChangeText={Function} // функция на изменение текста
onContentSizeChange={({nativeEvent: {contentSize: {width, height} }}) => void}
onEndEditing={Function} // коллбек на окончание ввода
//
onPressIn={({nativeEvent: PressEvent}) => void}
onPressOut={({nativeEvent: PressEvent}) => void}
onFocus={({nativeEvent: LayoutEvent}) => void}
onKeyPress={({nativeEvent: {key: keyValue} }) => void}
onLayout={({nativeEvent: LayoutEvent}) => void} // вызывается при монтировании компонента
onScroll={({nativeEvent: {contentOffset: {x, y} }}) => void}
onSelectionChange={({nativeEvent: {selection: {start, end} }}) => void}
onSubmitEditing={({nativeEvent: {text, eventCount, target}}) => void}
placeholder={'string'}
placeholderTextColor={'hexString'}
readOnly={false}
returnKeyType={'done'|'go'|'next'|'search'|'send'|'none'|'previous'|'default'|'emergency-call'|'google'|'join'|'route'|'yahoo'}
secureTextEntry={false}
selection={{start: number,end: number}}
selectionColor={'hexString'}
selectTextOnFocus={true}
showSoftInputOnFocus={true}
textAlign={'left'|'center'|'right'}
textContentType ={'none'|'addressCity'|'addressCityAndState'|'addressState'|'birthdate'|'birthdateDay'|'birthdateMonth'|'birthdateYear'|'countryName'|'creditCardExpiration'|'creditCardExpirationMonth'|'creditCardExpirationYear'|'creditCardFamilyName'|'creditCardGivenName'|'creditCardMiddleName'|'creditCardName'|'creditCardNumber'|'creditCardSecurityCode'|'creditCardType'|'emailAddress'|'familyName'|'fullStreetAddress'|'givenName'|'jobTitle'|'location'|'middleName'|'name'|'namePrefix'|'nameSuffix'|'newPassword'|'nickname'|'oneTimeCode'|'organizationName'|'password'|'postalCode'|'streetAddressLine1'|'streetAddressLine2'|'sublocality'|'telephoneNumber'|'URL'|'username'}

//Android props only
cursorColor={type Color} //цвет каретки
disableFullscreenUI={false}
importantForAutofill={'auto' | 'no' | 'noExcludeDescendants' | 'yes' | 'yesExcludeDescendants'}//
inputAccessoryViewID={String}
numberOfLines={0}
returnKeyLabel={String}
rows={0} //
textBreakStrategy={'simple'|'highQuality'|'balanced'}//перенос текста
underlineColorAndroid={type Color}

//iOS props only
clearButtonMode={'never'| 'while-editing'| 'unless-editing'| 'always'} //только для текст полей в одну линию
clearTextOnFocus ={true}//удаляет текст при наведении
contextMenuHidden={false} //скрывать контекстное меню
enablesReturnKeyAutomatically={false}//скрыть кнопке возврата
dataDetectorTypes={'phoneNumber', 'link', 'address', 'calendarEvent', 'none', 'all'} //интерактивность текста
inlineImageLeft={String} //изображение должно быть в /android/app/src/main/res/drawable inlineImageLeft='search_icon'
inlineImagePadding={0}//расстояние от inlineImageLeft
keyboardAppearance={'default'| 'light'| 'dark'} //цвет клавиатуры ввода
rejectResponderTermination={true}//передача событий выше
scrollEnabled={true}
spellCheck={true}
textContentType={}
passwordRules={String}
lineBreakStrategyIOS={'none', 'standard', 'hangul-word', 'push-out'}
// не поддерживается borderLeftWidth borderTopWidth borderRightWidth, borderBottomWidth, borderTopLeftRadius, borderTopRightRadius,
// borderBottomRightRadius, borderBottomLeftRadius,
style={type Style}
/>
```

методы

```js
.focus()
.blur()
clear()
isFocused()
```
