FROM --platform=linux/amd64 python:3.9

LABEL authors="Zeev_Rispler"

RUN pip install uv
COPY requirements.txt .
RUN uv venv -p python3.9 --seed -n /root/.venv  && \
    . /root/.venv/bin/activate && \
    uv pip install -r requirements.txt
COPY . .

RUN . /root/.venv/bin/activate && \
    python3 -m src.main initdb

RUN . /root/.venv/bin/activate && \
    python3 -m company_data.main

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]