From ubuntu:18.04
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
RUN apt-get install -y python3-pip
RUN pip3 install pandas
RUN pip3 install pyspark --no-cache-dir
COPY Project2.py Project2.py
COPY winequality-white.csv winequality-white.csv
CMD ["python3", "Project2.py"]
