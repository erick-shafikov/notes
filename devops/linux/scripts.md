# bash scripts

- #!/bin/bash - выбор интерпретатора
- echo - команда лога

```bash
#!/bin/bash

### This is comment ###

echo "welcome"
echo
echo "########################################"
echo "the uptime of the system is:"
uptime
echo

echo '########################################'
echo "memory:"
free -m
echo

echo '########################################'
echo "disk:"
df -h

```

пример запуска и сайта

```bash
#!/bin/bash

# установка httpd вывод всех логов в null
sudo yum install wget unzip httpd -y > /dev/null

# запуск httpd
sudo systemctl start httpd
sudo systemctl enable httpd

# для файлов сайта готовим директории
mkdir -p /tmp/webfiles
cd /tmp/webfiles

# скачивание сайта
wget https://www.tooplate.com/zip-templates/2098_health.zip > /dev/null
unzip 2098_health.zip > /dev/null
sudo cp -r 2098_health/* /var/www/html/

systemctl restart httpd
# удалить файлы
rm -rf /tmp/webfiles

# вывод системаной информации и файлов
sudo systemctl status httpd
ls /var/www/html/
~
```

# переменные

с использованием переменных

```bash
:!/bin/bash

# установка переменных
PACKAGE="httpd wget unzip"
SVC="httpd"
URL="https://www.tooplate.com/zip-templates/2098_health.zip"
APP_NAME="2098_health"
TEMPDIR="/tmp/webfiles"

# установка httpd вывод всех логов в null
sudo yum install $PACKAGE -y > /dev/null

# запуск httpd
sudo systemctl start $SVC
sudo systemctl enable $SVC

# для файлов сайта готовим директории
mkdir -p $TEMPDIR
cd $TEMPDIR

# скачивание сайта
wget $URL > /dev/null
unzip $APP_NAME.zip > /dev/null
sudo cp -r $APP_NAME/* /var/www/html/

systemctl restart $SVC
# удалить файлы
rm -rf $TEMPDIR

# вывод системаной информации и файлов
sudo systemctl status $SVC
ls /var/www/html/

```

# передача аргументов в скрипт

```bash
# файл args.sh
echo "Value of 0 is "
echo $0 # первый всегда имя файла

echo "Value of 1 is"
echo $1 # arg1

echo "Value of 2 is"
echo $2 # arg2

echo "Value of 3 is"
echo $3
```

```bash
./args.sh "arg1" "arg2"....
```

Системные переменные:

- $0 - имя скрипта
- $1-$9 - переменные скрипта
- $# - сколько аргументов передано в скрипт
- $@ - все аргументы
- $? - статус выхода последнего процесса, если 0 то последняя команда завершилась успешно, если отличная от 0, то завершилась с ошибкой
- $$ - id процесса
- $USER - имя пользователя
- $HOSTNAME - имя хоста
- $SECONDS - номер со 2го
- $RANDOM - произвольное число
- $LINENO - текущая строка скрипта

# кавычки

```bash
"" # - можно вставить переменную для спец символов $,... используют \$
'' # экранирует
```

# присвоение результата

Два варианта присвоения X=`вычисления` или X=$(вычисления)

```bash
CURRENT_USER=$(who)
echo $CURRENT_USER
# vagrant pts/0 2025-12-03 15:50 (10.0.2.2)
FREE_RAM=`free -m | grep Mem | awk '{print $4}'`
echo $FREE_RAM # 150
```

# user input

```bash
#!/bin/bash

echo "Enter"
read SKILL # ввод

echo "input: $SKILL"

read -p "username: " USR
read -sp "Password: " pass

echo "user: $USR"

```

# логические операции

```bash
if [<condition>]
then
<commands>
fi
```

```bash
read -p 'enter: ' NUM
echo

if [ $NUM -gt 100 ]
then
        echo 'in if block'
        sleep 3
        echo 'num is greater 100'
        echo
        date
else
        echo 'else block'
fi

echo 'complete'

```

вложенные условия

```bash
#!/bin/bash

ls /var/run/httpd/httpd.pd

date
ls /var/run/httpd/http.pid &> /dev/null

if [ $? -eq 0 ]
then
        echo 'httpd is running'
else
        echo 'httpd process is not running'
        echo 'start process'
        systemctl start httpd

        if [ $? -eq 0 ]
        then
                echo 'process starrted'
        else
                echo 'failed'
        fi
fi

```

# loop for

```bash
for VAR1 in java .net python ruby php # перебираемый объект
do
        sleep 1
        echo "Var1 is $VAR1"
        date
done

```

```bash
MYUSERS='alpha beta gamma'

for usr in $NYUSERS
do
        echo "adding $usr"
        useradd $usr
        id $usr
done

```

# loop while

```bash
counter=0

while [ $counter -lt 5 ]
do
        echo "loop"
        echo "value: $counter"
        counter=$(( $counter + 1))
done


# бесконечный
while
do
        echo "loop"
        echo "value: $counter"
        counter=$(( $counter + 1))
done


```

# remote
