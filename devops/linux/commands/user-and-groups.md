# группы и пользователи

Каждый пользователь имеет уникальный uuid
По умолчание присваивается home

- root id == 0
- обычный пользователь id > 1000
- сервисные пользователи

```bash
useradd ansible # добавление пользователей
useradd jenkins
useradd aws
groupadd devops # добавление группы пользователей
usermod -aG devops ansible
id ansible # получение id пользователей

vim /etc/group  # редактирование групп
passwd ansible # установка паролей
passwd aws
passwd jenkins
su - ansible # переключение
lsof -u vagrant
lsof -u aws
userdel aws # удаление
userdel -r jenkins
groupdel devops # удаление группы
```
