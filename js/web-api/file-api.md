```js
// Доступ к файлу (к единственному, если нет multiple)
const selectedFile = document.getElementById("input").files[0];
// доступ к файлам
const selectedFile = document.getElementById("input").files;
```

```js
const inputElement = document.getElementById("input");
inputElement.addEventListener("change", handleFiles, false);
function handleFiles() {
  const fileList = this.files;

  // количество файлов
  var numFiles = files.length;

  for (var i = 0, numFiles = files.length; i < numFiles; i++) {
    var file = files[i];
    // каждый файл имеет:
    file.name; //название файла
    file.size; //размер в байтах
    type; //MIME тип
  }
}
```

# drag&drop

```js
var dropbox;

dropbox = document.getElementById("dropbox");
dropbox.addEventListener("dragenter", dragEnter, false);
dropbox.addEventListener("dragover", dragOver, false);
dropbox.addEventListener("drop", drop, false);

function dragEnter(e) {
  e.stopPropagation();
  e.preventDefault();
}

function dragOver(e) {
  e.stopPropagation();
  e.preventDefault();
}

function drop(e) {
  e.stopPropagation();
  e.preventDefault();

  var dt = e.dataTransfer;
  var files = dt.files;

  handleFiles(files);
}
```

# Использование URL объектов для отображения изображений

```html
<input
  type="file"
  id="fileElem"
  multiple
  accept="image/*"
  style="display:none"
  onchange="handleFiles(this.files)"
/>
<a href="#" id="fileSelect">Select some files</a>
<div id="fileList">
  <p>No files selected!</p>
</div>
```

```js
window.URL = window.URL || window.webkitURL;

var fileSelect = document.getElementById("fileSelect"),
  fileElem = document.getElementById("fileElem"),
  fileList = document.getElementById("fileList");

fileSelect.addEventListener(
  "click",
  function (e) {
    if (fileElem) {
      fileElem.click();
    }
    e.preventDefault(); // prevent navigation to "#"
  },
  false
);

function handleFiles(files) {
  if (!files.length) {
    fileList.innerHTML = "<p>No files selected!</p>";
  } else {
    var list = document.createElement("ul");
    for (var i = 0; i < files.length; i++) {
      var li = document.createElement("li");
      list.appendChild(li);

      var img = document.createElement("img");
      // получаем URL файла
      img.src = window.URL.createObjectURL(files[i]);
      img.height = 60;

      img.onload = function () {
        window.URL.revokeObjectURL(this.src);
      };

      li.appendChild(img);
      var info = document.createElement("span");
      info.innerHTML = files[i].name + ": " + files[i].size + " bytes";
      li.appendChild(info);
    }
  }
}
```

# Загрузка

```js
function FileUpload(img, file) {
  const reader = new FileReader();
  this.ctrl = createThrobber(img);
  const xhr = new XMLHttpRequest();
  this.xhr = xhr;

  const self = this;

  this.xhr.upload.addEventListener(
    "progress",
    function (e) {
      if (e.lengthComputable) {
        const percentage = Math.round((e.loaded * 100) / e.total);
        self.ctrl.update(percentage);
      }
    },
    false
  );

  xhr.upload.addEventListener(
    "load",
    function (e) {
      self.ctrl.update(100);
      const canvas = self.ctrl.ctx.canvas;
      canvas.parentNode.removeChild(canvas);
    },
    false
  );

  xhr.open(
    "POST",
    "https://demos.hacks.mozilla.org/paul/demos/resources/webservices/devnull.php"
  );

  xhr.overrideMimeType("text/plain; charset=x-user-defined-binary");
  reader.onload = function (evt) {
    xhr.send(evt.target.result);
  };

  reader.readAsBinaryString(file);
}
```

```js
// обработка pdf
var obj_url = window.URL.createObjectURL(blob);
var iframe = document.getElementById("viewer");
iframe.setAttribute("src", obj_url);
window.URL.revokeObjectURL(obj_url);
// обработка видео
var video = document.getElementById("video");
var obj_url = window.URL.createObjectURL(blob);
video.src = obj_url;
video.play();
window.URL.revokeObjectURL(obj_url);
```
