import json
from llm_client import ask_gigachat  # БЫЛО "from LLM_CLIENT" — проверьте регистр имени файла!


def classify_news_prompt(text: str) -> str:
    """Промпт для классификации новости по тематике (под ваш датасет)."""
    TOPICS = ["Экономика", "Политика", "Спорт", "Технологии", "Культура", "Наука", "Происшествия"]
    prompt = f"""
    Ты — ассистент для классификации новостных текстов по тематике.
    Выбери ровно ОДНУ тему из закрытого списка: {", ".join(TOPICS)}.

    Текст новости: "{text}"

    Ответь СТРОГО валидным JSON без markdown-обёрток (без ```json), со структурой:
    {{
        "topic": "одна тема из списка",
        "confidence": 0.0,
        "reason": "краткое обоснование выбора одной фразой"
    }}
    """
    return prompt


if __name__ == "__main__":
    # Тестовый текст НОВОСТИ (не отзыв!) из вашего датасета
    user_text = """
    Центральный банк повысил ключевую ставку до 21%. Аналитики прогнозируют
    ослабление рубля и рост инфляции в ближайшем квартале.
    """

    print("1. Формируем промт для классификации топика...")
    final_prompt = classify_news_prompt(user_text)

    print("2. Отправляем промт в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1)

    print(f"\nСырой ответ от модели:\n{raw_response}\n")

    print("3. Проверка валидности полученного JSON...")
    try:
        # Очищаем ответ от markdown-обёртки ```json ... ```, если модель её добавила
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.removeprefix("json").strip()

        parsed_json = json.loads(cleaned)  # БЫЛО "parced_json" — опечатка
        print("Успех! Данные преобразованы в python dict:")
        print(f"Топик: {parsed_json.get('topic')}")          # БЫЛО 'sentiment'
        print(f"Уверенность: {parsed_json.get('confidence')}") # БЫЛО 'pros'
        print(f"Обоснование: {parsed_json.get('reason')}")     # БЫЛО 'cons'
    except json.JSONDecodeError:
        print("Ошибка: Модель вернула невалидный JSON")