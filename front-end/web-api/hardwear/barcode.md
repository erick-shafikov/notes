# barcode detector (-ch, -e, -ff, -s)

сканирование штрих кодов, qr..

статические методы

- getSupportedFormats()

методы экземпляра

- detect(imageBitmapSource) ⇒ промис
- - imageBitmapSource: HTMLImageElement, SVGImageElement, HTMLVideoElement, HTMLCanvasElement, ImageBitmap, OffscreenCanvas, VideoFrame, Blob

```js
// check compatibility
if (!("BarcodeDetector" in globalThis)) {
  console.log("Barcode Detector is not supported by this browser.");
} else {
  console.log("Barcode Detector supported!");

  // create new detector
  const barcodeDetector = new BarcodeDetector({
    formats: ["code_39", "codabar", "ean_13"],
  });
}

barcodeDetector
  .detect(imageEl)
  .then((barcodes) => {
    barcodes.forEach((barcode) => console.log(barcode.rawValue));
  })
  .catch((err) => {
    console.error(err);
  });
```
