#FROM docker.io/ubuntu:20.04
#RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

FROM quay.balgroupit.com/baloise-base-images/python:3-ubuntu20.04

WORKDIR /work

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
