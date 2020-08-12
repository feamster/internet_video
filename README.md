# internet_video

## Start Webservice
    pip install flask
    pip install youtube-dl
    python3 server.py

## Run codes in models

### Install Linear Algebra Package 

#### Ubuntu (>=14.04) or Debian (>=7)
    sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev

#### macOS
    brew install openblas

#### Windows
   download `LAPACK` `LAPACKE` `BLAS` from https://icl.cs.utk.edu/lapack-for-windows/lapack/. Then, put them in the windows library folder (default: `C://Windows/System32`)


### Install python libraries
    setuptools
    numpy
    scipy
    scikit-learn<=0.19.2
    matplotlib
    Cython
    joblib

### Install libact
    pip3 install git+https://github.com/hyperpro/libact.git

### Install modAL

modAL requires:

```
python >= 3.5
numpy >= 1.13
scipy >= 0.18
scikit-learn >= 0.18
```

Install modAL directly with pip:

```
pip install modAL
```

### Install MongoDB

Install packages:

```
wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org=4.2.5 mongodb-org-server=4.2.5 mongodb-org-shell=4.2.5 mongodb-org-mongos=4.2.5 mongodb-org-tools=4.2.5
sudo systemctl daemon-reload
sudo systemctl start mongod
```

To verify MongoDB has started:

```
sudo systemctl status mongod
```

To ensure MongoDB will start following a system reboot:

```
sudo systemctl enable mongod
```

Create user and shutdown MongoDB instance:

```
mongo
use admin
db.createUser({user:"admin",pwd:"63czhj6c8pe",roles:[{role:"userAdminAnyDatabase",db:"admin"},"readWriteAnyDatabase"]})
db.createUser({user:"uchivideo1",pwd:"63czhj6c8pe",roles:[{role:"userAdminAnyDatabase",db:"admin"},"readWriteAnyDatabase"]})
db.adminCommand({shutdown:1})
quit()
```

Add the `security.authorization` setting to `/etc/mongod.conf` :

```
security:
  authorization: enabled
```

Restart MongoDB:

```
sudo systemctl start mongod
```

Connect with authentication:

```
mongo -u admin -p 63czhj6c8pe localhost:27017/admin
```

Create collections:

```
db.createCollection("record")
db.createCollection("users")
```

