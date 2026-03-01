FROM python:3.12-slim

RUN pip install --no-cache-dir srclight

ENTRYPOINT ["srclight"]
