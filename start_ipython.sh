# Assume the virtualenv is called .env

cp frameworkpython .env/bin
chmod a+x .env/bin/frameworkpython
.env/bin/frameworkpython -m jupyter notebook
