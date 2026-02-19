FROM python:3.12-slim

WORKDIR /app

# No external deps needed - pure stdlib
COPY *.py ./

# Create data directory for SQLite
RUN mkdir -p /data

ENV CLAWBIZARRE_DB=/data/clawbizarre.db
ENV CLAWBIZARRE_PORT=8420

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health')" || exit 1

CMD ["python3", "api_server_v6.py", "--port", "8420", "--db", "/data/clawbizarre.db"]
