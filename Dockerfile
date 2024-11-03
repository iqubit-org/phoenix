FROM ubuntu:22.04

# install Python 3.10 and pip
RUN apt-get update
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip


# set alias of python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# verify installation
RUN python --version
RUN pip --version

# set the working directory in the container
WORKDIR /app

# copy the current directory contents into the container at /app
COPY . /app

# install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# run the command to start up the application
CMD ["/bin/bash"]

