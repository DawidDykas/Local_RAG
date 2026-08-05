FROM python:3.10

WORKDIR /app

EXPOSE 5000

COPY . .

RUN pip install --upgrade pip
RUN pip config list
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]