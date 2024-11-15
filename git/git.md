# основные команды git

- **git init** – инициализировать проект
- **git status** – получить информацию о проекте
- **git commit -m "Complete Chapter 1“** – добавить изменения с пояснением (в настоящем времени)
- **git add .** – добавить все изменения в директории проекта
- **git log** – информация о проекте на текущий момент
- **git diff chapter3.txt** – узнать об изменениях в файле
- **git checkout chapter3.txt** – вернуть изменения, относительно последнего commit
- **git remote add origin https://github.com/erick-shafikov/Story.git** - запустить проект на github
- **git push -u origin master** – добавляем на мастер
- **git rm --cached -r .** - удаляем файлы из промежуточного репозитория
- **git clone https://....** – скопировать

# gitignore

В первую очередь нужно создать файл .gitignore

Создать файл -.gitignore
#Comment - комментарии
\*.txt – игнорирование по разрешению файла

# branch

- **git branch branch-name** – создание нового ответвления
- **git branch** – просмотр направления master – ответвления, в консоли будет отображено:
  ** \*branch-name** //звездочка в начел указывает на выбранное направление
  master //не выбран
- **git checkout branch-name (commit)** – переключится на branch-name ответвление или коммит, если добавить названия файлов в конце, то откатятся файлы
- **git checkout** –f branch-name – переключится на branch-name со сбросом изменений (-f === -force)
- **git checkout** -f HEAD - удалить
- **git stash** – переключится на branch со сбросом изменений
- **git checkout master** – переходим на мастер-ветку
- **git checkout branch-name** – возвращаемся
- **git stash pop** – возвращаем все изменения на любую ветку

git checkout –f – удалить все изменения

## pull request

- Fork – скопировать к себе проект
- Create pull request – предложить изменению
- merge pull request – принять изменения

## git show

Позволяет посмотреть версию файла несколько коммитов назад
git show @~~ == git show HEAD~~

## git merge

- git merge <branch-name-to-merge> - если мы находимся на некоторой ветке, то она сольется с веткой branch-name-to-merge
- git merge fe<branch-name-to-merge> -

семантические конфликты – это конфликты, которые приводят к ошибкам

- git merge –-squash – удаляет всю историю коммитов без связки с веткой
- git merge –-abort – отмена слияния
- git reset –-merge – отмена слияния

## git reset

- git reset –-hard commit - сброс до старого коммита, если нет <commit> , <commit> === HEAD
- git reset --soft commit - сброс до старого коммита, но изменения в коммитах остаются

## git commit

- git commit –-amend – git commit --soft
- git commit –am ‘message’ === git add . + git commit ‘message’

## git ceckout

- git checkout –ours – при слиянии показывает нашу версию комиита
- git checkout –ours – при слиянии показывает нашу версию комиита
- git checkout –-conflict=diff3 –merge index.html – при конфликте откатится к базовому коммиту (от которого наследовались две конфликтующие ветки)
  ----- далее исправляем код (изменяем относительно двух фалов ручками) ------
- git show :1 , :2, :3 – показывает версии предка, нашу и чужую версию файла
- git add –
- git merge –-continue === git commit в случае слияния

## git rebase

- git rebase master – перенос ветки, перенос HEAD
- git rebase –-abort – отмена
- git rebase –-skip – не замечать конфликты
- git rebase –-x <команда на выполнение node index.js>– проверка на ошибки при rebase
- git rebase onto <base branch> <source branch> - перенос ответвлениеё
- git rebase –-rebase-merges <branch> - слияние

## git revert

git revert – отмена коммитов

## Удаление коммита

Удалить его локально
git reset HEAD~1 --hard

Сделать force push на сервер
git push -f

# Варианты ветвления

- Версионные
- Тематические

## cherry picking

HEAD – указкатель на коммит, от которого идут текущие изменения
Возможно применять какой-либо коммит к другой ветке

git cherry-pick D – позволяет копировать коммиты, вносит изменения в эквивалентные коммиты, то есть вносит diff из одной ветки в другую

git cherry-pick -x – показывает разницу

git cherry-pick "branch name" master - копирование коммитов из одного в другой

git cherry-pick –-abort – отмена cherry-pick
git cherry-pick –-n – копирование коммита без применения
