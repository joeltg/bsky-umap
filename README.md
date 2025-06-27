# bsky-umap



## EC2 setup

```sh
sudo yum update -y
sudo yum groupinstall "Development Tools"
sudo yum install -y \
    git gcc zlib-devel bzip2-devel readline-devel \
    sqlite-devel openssl-devel tk-devel libffi-devel xz-devel \
    tmux htop

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source .bashrc

pyenv install 3.12.3
pyenv global 3.12.3

git clone https://github.com/joeltg/bsky-umap.git
cd bsky-umap
git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

configure a persistent data volume mount:

```sh
sudo mkdir /data
echo '/dev/sdb /data ext4 defaults 0 2' | sudo tee -a /etc/fstab
sudo mount -a
sudo mount /dev/sdb /data
```

```sh
mkdir /data/${SNAPSHOT}
touch /data/${SNAPSHOT}/graph.sqlite
touch /data/${SNAPSHOT}/directory.sqlite
touch /data/${SNAPSHOT}/ids.buffer
```

## misc

to hash a `tiles/` directory:

```
find tiles -type f -exec sha256sum {} \; | sort | sha256sum
```
