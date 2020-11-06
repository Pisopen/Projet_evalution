FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt

Expose 8501

CMD streamlit run /app/scripts/algoRF.py