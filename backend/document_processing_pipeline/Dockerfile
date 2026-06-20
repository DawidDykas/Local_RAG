FROM python:3.8

WORKDIR /app

COPY . .


EXPOSE 5000
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]