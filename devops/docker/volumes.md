# volumes

При удалении контейнера данные удаляются. volumes позволяют создать директорию с данными. Альтернатива bind mounts

```bash
docker run --name some-sql -v /my/own/datadir/:/var/lib/mysql # перенаправление в случае sql из /my/own/datadir/ в предустановленную /var/lib/mysql
docker volume create data-name # создать volume
docker volume rm # удаление volume
docker volume prune # удаление volume
```

docker volume #
