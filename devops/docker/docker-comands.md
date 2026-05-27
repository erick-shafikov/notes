# образы

```bash
docker pull <image-name> # скачать image
docker pull <image-name>:<tag> # скачать image c определенным тегом
docker images # все образы
docker images rmi # удалить образы
docker stop <container-name> # остановить контейнер по имени
```

# контейнер

```bash
docker ps # все активные процессы
docker ps -a # все процессы активные и нет
docker run <container-name> # запустить container-name
docker start/stop/restart/rm # действия с контейнером
docker run --name -d <container-name> # запустить image container-name
# флаги
# --name - имя,
# -d - detach режим, будет активном в терминале
# -p 7090:80 - 7090 порт хоста, 80 - порт в контейнере
# -P - автоматический матч портов
# -e SOME_VAR=VAR_VAL # для передачи переменных

```

# работа внутри контейнера

```bash
cd /vat/lib/docker # где располагается докер
cd /vat/lib/docker/containers/container-hash # где располагается контейнер
docker exec <container-name-or-id> # покажет контейнер
docker exec -id <container-name-or-id> /bin/bash # войти в контейнер
docker inspect # json - конфигурация контейнера, метадата
# есть cmd-команда в составе показывает какой скрипт запускается

docker logs # логи, сами по себе это вывод процесса

```

# образ

```bash
docker build # создание образа из докер файла
# -t image-name - имя образа
docker system prune -a # удалить все
```

docker #
