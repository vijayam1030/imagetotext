# Image to Text Extractor

A simple web application that extracts text from images using LLaVA vision model through Ollama.

## Prerequisites

1. **Ollama installed** - Make sure Ollama is running on your system
2. **LLaVA model** - Download the LLaVA model:
   ```bash
   ollama pull llava
   ```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Upload an image and click "Extract Text" to get the text content

## Features

- **Simple UI** - Easy-to-use web interface
- **Multiple formats** - Supports PNG, JPG, JPEG, GIF, BMP
- **Model selection** - Choose from different LLaVA model sizes
- **Text extraction** - Extract text from images or get image descriptions
- **Copy functionality** - Easy copying of extracted text

## Troubleshooting

- Make sure Ollama is running: `ollama serve`
- Verify LLaVA model is installed: `ollama list`
- Check if the model name matches what you have installed