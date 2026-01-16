#!/usr/bin/env bash
set -e

echo "▶ Обновление системы"
apt update -y

echo "▶ Установка Python и утилит"
apt install -y python3 python3-venv python3-pip git curl

echo "▶ Создание виртуального окружения"
python3 -m venv venv
source venv/bin/activate

echo "▶ Установка зависимостей"
pip install --upgrade pip
pip install google-generativeai python-dotenv fastapi uvicorn

echo "▶ Готово"
