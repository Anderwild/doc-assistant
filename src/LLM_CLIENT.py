import os

from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

load_dotenv()


def ask_gigachat(prompt: str, temperature: float = 0.3) -> str:
    """Отправляет текстовый промпт в GigaChat и возвращает ответ."""
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError(
            "Ошибка: переменная GIGACHAT_CREDENTIALS не найдена в файле .env"
        )

    # verify_ssl_certs=False отключает проверку SSL-сертификата.
    with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
        response = giga.chat(
            Chat(
                model="GigaChat-3-Lightning",
                temperature=temperature,
                max_tokens=1000,
                messages=[
                    Messages(
                        role=MessagesRole.USER,
                        content=prompt,
                    )
                ],
            )
        )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("Проверка связи с GigaChat...")
    test_question = "Что такое промпт-инжиниринг в трёх предложениях?"

    try:
        answer = ask_gigachat(test_question)
        print(f"\nВопрос: {test_question}")
        print(f"Ответ:\n{answer}")
    except Exception as error:
        print(f"Произошла ошибка при подключении: {error}")

