# **Gradio Webpage Chatbot**

Gradio Webpage Chatbot is a Python application that allows you to generate context and have a conversation with a chatbot based on web content. It uses Gradio for the user interface and leverages Langchain and Chroma for text and content retrieval.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

- Python 3.x
- Gradio
- Langchain
- Chroma
- PyMuPDF
- PIL (Pillow)
- requests

You will also need an OpenAI API key to enable chat capabilities.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/gradio-webpage-chatbot.git
cd gradio-webpage-chatbot
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key in the `keys.py` file.

## Usage

1. Run the Gradio application:

```bash
python gradio_webpage_chatbot.py
```

2. Access the application in your web browser by opening the provided URL.

3. Enter a query in the "URL WebPage Chatbot" section and specify the number of URLs you want to retrieve content from.

4. Click the "Generate context" button to generate context from the web content.

5. Enter your text in the "Enter your text here and press submit" text box.

6. Click the "submit" button to interact with the chatbot. The chat history and generated PDF will be displayed on the right side of the application.

## Screenshots

![Screenshot 2](screenshots/image-1.png)

![Screenshot 1](screenshots/image.png)


![Screenshot 3](screenshots/image-2.png)

![Screenshot 4](screenshots/image-3.png)

