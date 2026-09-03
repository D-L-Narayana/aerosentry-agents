FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/make_imagery.py && python -m pytest -q
ENV AEROSENTRY_PROVIDER=mock
EXPOSE 8000
CMD ["uvicorn", "aerosentry.api:app", "--host", "0.0.0.0", "--port", "8000"]
