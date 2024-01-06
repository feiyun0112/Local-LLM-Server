import sys
import os

# 日志存储路径
LOG_PATH = os.path.join(os.path.abspath(os.curdir), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

HOST = "127.0.0.1"


API_SERVER_CONFIG = {
    "host": HOST,
    "port": 21000,
    "api_keys": [],
}

CONTROLLER_CONFIG = {
    "host": HOST,
    "port": 21001,
}

WORK_CONFIG = {
    "host": HOST,    
    "port": 21002,
    "models": {
        "ChatModel":"d:/chatglm3-6b",
        "EmbeddingsModel":"./models/bge-large-zh",
    },    
}
