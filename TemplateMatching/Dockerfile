FROM python:3.10.10

WORKDIR /usr/src/app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install libgl1
CMD ["python", "main.py", "-n", "1", "manual"]


