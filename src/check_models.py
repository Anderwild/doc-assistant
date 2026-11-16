import os
from dotenv import load_dotenv
from gigachat import GigaChat
load_dotenv()
if __name__=="__main__":
    with GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"),verify_ssl_certs=False) as giga:
        models_data=giga.get_models()
        for m in models_data.data:
            print(f"доступная модель: {m.id_}")
