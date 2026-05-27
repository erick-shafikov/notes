# Linux

# команды

```bash
# command options arguments
  ls      -l      /tmp/

command --help #даст справку по команде
```

```bash

whoami # показать пользователя
pwd # нахождение
ls # список файлов
ls -lt # сортировка по времени
ls -ltк # поменять в обратном порядке
cat # открыть файл


mkdir dev #создать файл
mkdir ops backupdir #создать несколько файлов
mkdir -p /opt/dev/ops/devops/test #создать иерархию

# backupdir dev ops
touch testfile.txt # создать файл пустой
touch deviosfile{1..10} # создать набор файлов
# CP копирование файлов
cp devopsfile1.txt dev/ # копировать из текущей в dev
cp dev/devopsfile1.txt dev/ # копировать из текущей в dev по абсолютной ссылке
cp -r # для копирования директорий
# MV перемещение файлов

# MV команды
mv file dir/ # переместить file в dir/
mv file file1 # переименовать file в file1
mv *.txt textdir/

# rm
rm deviosfile10 # удаление файлов
rm -r dir/ # удаление директорий
rm -r * # удалить все
```
