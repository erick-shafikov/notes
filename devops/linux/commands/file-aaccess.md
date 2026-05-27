# файлы

в linux все файл

```bash
ls -l # посмотреть тип файла
-rw-------. 1 root root 2027 Oct 10 2024 anaconda-ks.cfg # если начинается с "-" то это обычный файл
drwxr-xr-x. 2 root root 6 Oct 26 17:26 devopsdir
-l---- # link
-d---- # directory
```

# доступы к файлам

```bash
ls -l # информация

# read and write, user - root, group - root
-rw-------. 1 root root 2058 Oct 26 18:17 anaconda-ks.cfg
# для root и для других
drwxr-xr-x. 2 root root   31 Oct 26 18:30 devopsdir
-rw-------. 1 root root 1388 Oct 10  2024 original-ks.cfg
# r -read, w - write, x - execute
```

```bash
chmod -x # разрешить исполнение
chmod o-x /opt/devopsdir # o-x => others execute
chmod o-r /opt/devopsdir # o-r => others read

ls -ld /opt/devopsdir # проверка доступа
chmod g+w /opt/devopsdir # для групп
ls -ld /opt/devopsdir
chown aws.devops /opt/webdata
chmod -R 770 /opt/webdata # определение прав с помощью номеров
chmod -R 754 /opt/webdata
```
