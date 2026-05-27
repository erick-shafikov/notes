# процессы

Процессы выстраиваются в дерево процессов начиная с процесса init, с id == 1 и каждый процесс вызывает другой

делятся на:

- мертвые процессы
- зомби процесс
- активные процесс

```bash
ps -ef | grep httpd # проверка процессов с фильтром
kill -9 1476 # убить процесс
kill -9 5772 # может быть и дочерний процесс
ps -ef | grep httpd | grep -v 'grep' | awk '{print $2}' # отобразить только второй столбец
ps -ef | grep httpd | grep -v 'grep' | awk '{print $2}' | xargs kill -9 # отобразить только второй столбец и убить все процессы
```

# глобальные переменные

находятся:

- /.profile
- /.bashrc
- /etc/profile - для всех

```bash
SEASON='Monsoon' # установим переменную

exit # установим переменную
echo $SEASON # она потеряет значение
# создадим в файле testvars переменную
cd /opt/scripts/
vim testvars.sh
chmod +x testvars.sh
# экспортируем из файла
./testvars.sh
export SEASON

# если нужна глобальная в .bashrc
source .bashrc

# если нужна для профиля в /etc/profile

vim /etc/profile
# добавить строчку
# export SEASON='winter'

echo $SEASON
```
