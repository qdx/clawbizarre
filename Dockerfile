FROM python:3.12-slim

# For Docker-in-Docker sandboxing (optional)
# If you need language-agnostic execution (JS/Bash), mount the Docker socket
# -v /var/run/docker.sock:/var/run/docker.sock

WORKDIR /app

# Copy prototype (stdlib only, no pip install needed)
COPY prototype/ ./prototype/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8700/health')" || exit 1

EXPOSE 8700

# Default: hardened server, no auth (add --api-key for production)
ENTRYPOINT ["python3", "prototype/verify_server_hardened.py"]
CMD ["--host", "0.0.0.0", "--port", "8700"]
