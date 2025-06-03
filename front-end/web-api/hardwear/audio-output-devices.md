# audio output devices

позволяет управлять способом вывод звука с помощью метода MediaDevices.selectAudioOutput()

```js
document.querySelector("#myButton").addEventListener("click", async () => {
  if (!navigator.mediaDevices.selectAudioOutput) {
    console.log("selectAudioOutput() not supported or not in secure context.");
    return;
  }

  // Display prompt to select device
  const audioDevice = await navigator.mediaDevices.selectAudioOutput();

  // Create an audio element and start playing audio on the default device
  const audio = document.createElement("audio");
  audio.src = "https://example.com/audio.mp3";
  audio.play();

  // Change the sink to the selected audio output device.
  audio.setSinkId(audioDevice.deviceId);
});
```

# setSinkId

id - вывода звука
