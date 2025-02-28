FROM python:3.9.20

# Install dependencies
RUN pip install --no-cache-dir \
    requests \
    pydantic \
    xgboost \
    torch \
    transformers \
    pandas

COPY . .

CMD ["sh", "run.sh"]
