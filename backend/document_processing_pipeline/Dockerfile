FROM python:3.8

WORKDIR /app

COPY . .

# req in uv for fastapi and uvicorn


EXPOSE 5000
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]