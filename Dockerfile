FROM python:3.10

RUN mkdir -p /app/


COPY ./predict.py /app

COPY templates /app/templates

COPY requirements.txt /app/requirements.txt

COPY saved_model /app/saved_model

WORKDIR /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

    

EXPOSE 9692


ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9692", "predict:app" ]


