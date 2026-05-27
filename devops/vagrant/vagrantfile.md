```ruby
# config можно поменять на любое другое слово
Vagrant.configure("2") do |config|
  # подключение "коробок"
  config.vm.box = "ubuntu/jammy64"
  # или config.vm.box = "eurolinux-vagrant/centos-stream-9"
  # создание ip
  config.vm.network "private_network", ip: "192.168.56.14"
  # доступ к сети
  config.vm.network "public_network"
  # скопировать в vagrant/ директорию на ВМ
  config.vm.synced_folder "D:\\scripts\\shellscripts", "/opt/scripts"
  config.vm.provider "virtualbox" do |vb|
  #   кастомизация железа
    vb.memory = "1600"
    vb.cpus = "2"
  end
  # provision - команды выполняемые при установке машины (но не при загрузке) в ubuntu запустятся сразу
  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y apache2
  SHELL
end
```

# multi vagrant file

```ruby
Vagrant.configure("2") do |config|
    config.vm.define "web01" do |web01|
      web01.vm.box = "ubuntu/focal64"
      web01.vm.hostname = "web01"
      web01.vm.network "private_network", ip: "192.168.56.41"
    end

    config.vm.define "web02" do |web02|
      web02.vm.box = "ubuntu/focal64"
      web02.vm.hostname = "web02"
      web02.vm.network "private_network", ip: "192.168.56.42"
    end

    config.vm.define "web03" do |web03|
        web03.vm.box = "ubuntu/focal64"
        web03.vm.hostname = "web02"
        web03.vm.network "private_network", ip: "192.168.56.43"
      end

    config.vm.define "db01" do |db01|
      db01.vm.box = "eurolinux-vagrant/centos-stream-9"
      db01.vm.hostname = "db01"
      db01.vm.network "private_network", ip: "192.168.56.44"
      db01.vm.provision "shell", inline: <<-SHELL
        yum install -y wget unzip mariadb-server -y
        systemctl start mariadb
        systemctl enable mariadb
        # Additional provisioning steps for db01
      SHELL
    end
  end

```

```bash
vagrant ssh web01 # что бы зайти в конкретную вм
vagrant halt web01 # выключить определенную, если не указать, то остановит все
```

# боксы

https://portal.cloud.hashicorp.com/vagrant/discover/
