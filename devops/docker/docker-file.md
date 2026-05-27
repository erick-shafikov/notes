# docker file

# FROM

базовый образ

# LABELS

метадата

# RUN

запуск файлов

# ADD/COPY

добавить файлы и директории в образ

# CMD

запуск бинарников

```Dockerfile
CMD ["echo", "hello"] # echo - команда, hello - параметр
```

# ENTRYPOINT

конфигурация контейнера

```Dockerfile
ENTRYPOINT ["echo"] # echo - команда, параметры нужно передавать
```

ENTRYPOINT + CMD

```Dockerfile
ENTRYPOINT ["echo"]
CMD ["hello"]
```

# VOLUME

создает volumes

# EXPOSE

настройка контейнеров под порты для сетевой работы

# ENV

переменные окружения

# USER

установка имени

# WORKDIR

установка директории

# ARG

проброс аргументов во время билда

# ONBUILD

отложенные процедуры
