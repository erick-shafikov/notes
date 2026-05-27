# сервисы

запускают процессы

```bash
systemctl status httpd # статус сервиса
systemctl enable httpd # включение сервиса
systemctl is-active httpd # статус сервиса
systemctl is-enabled httpd # статус сервиса
# файл управления статусом
cat /etc/system/multi-user.target.wants/httpd.service
cat /etc/systemd/multi-user.target.wants/httpd.service
cat /etc/systemd/system/multi-user.target.wants/httpd.service

```
