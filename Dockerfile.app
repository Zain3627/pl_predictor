FROM python:3.11-slim

WORKDIR /app

RUN pip install streamlit pandas psycopg2-binary requests python-dotenv

COPY streamlit_app.py .
COPY .streamlit/ .streamlit/
COPY icons/ icons/

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]