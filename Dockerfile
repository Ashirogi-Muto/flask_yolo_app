FROM python:3
RUN mkdir /app/
COPY ["./app.py", "./coco.names", "./requirements.txt", "./yolov3-tiny.cfg", "./yolov3-tiny.weights", "./app/"]
WORKDIR /app
RUN pip install -r requirements.txt
CMD [ "python", "app.py" ]
EXPOSE 5000