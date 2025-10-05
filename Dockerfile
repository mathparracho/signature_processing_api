FROM python:3.10-slim

# Evita prompts interativos e reduz tamanho
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=1.8.3
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema necessárias (ex: OpenCV, Pillow)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar apenas os arquivos do Poetry primeiro (para aproveitar cache)
COPY pyproject.toml poetry.lock* ./

# Instalar Poetry e dependências do projeto
RUN pip install --upgrade pip \
    && pip install "poetry==$POETRY_VERSION" \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copiar o restante do código (app, processing, etc.)
COPY . .

# Expor a porta usada pelo Flask/Gunicorn
EXPOSE 5000

# Comando padrão (usa Gunicorn com 4 workers)
CMD ["poetry", "run", "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
