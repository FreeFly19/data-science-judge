FROM python:3.9
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY main.py /app
COPY /templates /app/templates

ENTRYPOINT ["python", "main.py"]