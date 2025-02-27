FROM python:3

# add - xgboost for the ML model
#     - torch + transformers for BERT
RUN pip install --no-cache-dir \
    requests \
    pydantic \
    xgboost \
    torch \
    transformers \
    spacy \
    pandas

# Optional
RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["sh", "run.sh"]
