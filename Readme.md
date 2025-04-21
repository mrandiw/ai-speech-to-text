![PythonAnywhere](https://img.shields.io/badge/pythonanywhere-%232F9FD7.svg?style=for-the-badge&logo=pythonanywhere&logoColor=151515)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)


# Python With OpenAI Whisper Implementation

## Overview

Whisper is an advanced automatic speech recognition (ASR) system developed by OpenAI, trained on 680,000 hours of multilingual and multitask supervised data. This repository contains a practical implementation of the Whisper model for audio transcription using Python. The project provides a user-friendly Gradio interface that allows users to easily transcribe audio files without writing code, making OpenAI's powerful speech recognition technology accessible to everyone.

### Key Features

- **Multilingual Support**: Recognizes and processes speech in multiple languages
- **High Accuracy**: Performs robustly across various accents and technical terminology
- **Translation Capability**: Supports translation between numerous languages
- **Customizable**: Offers flexible transcription options to suit different needs

## Original Project Reference
Official repository: [https://github.com/openai/whisper](https://github.com/openai/whisper)

## Requirements
- Python 3.12

## Installation

### 1. Environment Preparation
Create and activate a Python virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate

# Ubuntu
python -m venv venv
source venv/bin/activate
```

### 2. Install Required Dependencies

```bash
pip install -U openai-whisper
pip install gradio
```

## Usage

To launch the application, run:

```bash
python main.py
```

This will start the Gradio interface for audio transcription using the Whisper model.

## Downloaded Model Located in 
```bash
C:\Users\User\.cache\whisper
```