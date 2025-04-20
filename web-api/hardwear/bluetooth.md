Интерфейсы:

- Bluetooth
- - свойства:
- - - referringDevice - Возвращает ссылку на устройство
- - события:
- - - onavailabilitychanged - запускается при возникновении события availabilitychanged
- - методы
- - - getAvailability() ⇒ промис-boolean поддерживает ли браузер работу с bt
- - - requestDevice() ⇒ Promise объекту BluetoothDevice

- BluetoothCharacteristicProperties
- - свойства:
- - - authenticatedSignedWrites ⇒ bool знаковая запись в значение характеристики
- - - broadcast ⇒ bool
- - - indicate ⇒ bool
- - - notify ⇒ bool
- - - read ⇒ bool
- - - reliableWrite ⇒ bool
- - - writableAuxiliaries ⇒ bool
- - - write ⇒ bool
- - - writeWithoutResponse ⇒ bool

- BluetoothDevice - данные об устройстве
- - свойства:
- - - id
- - - name
- - - gatt
- - методы:
- - - watchAdvertisements()
- - - forget()

- BluetoothRemoteGATTCharacteristic - до данные
- - свойства:
- - - service
- - - uuid
- - - properties
- - - value
- - методы
- - - getDescriptor() ⇒ дескриптора UUID
- - - getDescriptors()
- - - readValue() ⇒ Promise DataView
- - - writeValue()
- - - writeValueWithResponse()
- - - writeValueWithoutResponse()
- - - startNotifications()
- - - stopNotifications()

- BluetoothRemoteGATTDescriptor
- BluetoothRemoteGATTServer
- BluetoothRemoteGATTService
- BluetoothUUID

Navigator.bluetooth ⇒ Bluetooth

```js
const btPermission = await navigator.permissions.query({ name: "bluetooth" });
if (btPermission.state !== "denied") {
  // Do something
}
```
