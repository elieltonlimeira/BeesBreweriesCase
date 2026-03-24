FROM apache/airflow:2.9.1-python3.11

USER root

# Install Java JRE (Spark only needs the runtime, not the full JDK — saves ~100MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3

# Download Hadoop AWS JARs required for s3a:// filesystem support in PySpark.
# Without these, any spark.read/write to s3a:// raises ClassNotFoundException.
ENV SPARK_VERSION=3.5.1
ENV HADOOP_AWS_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262
ENV SPARK_JARS_DIR=/opt/spark/jars

RUN mkdir -p ${SPARK_JARS_DIR} && \
    curl -fsSL -o ${SPARK_JARS_DIR}/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar" && \
    curl -fsSL -o ${SPARK_JARS_DIR}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
      "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

ENV SPARK_EXTRA_JARS_DIR=${SPARK_JARS_DIR}

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
