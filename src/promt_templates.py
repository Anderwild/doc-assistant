import json 
from LLM_CLIENT import ask_gigachat

def analyze_review_promt(review_text:str) -> str:
    prompt = f"""
    You are a helpful assistant that analyzes product reviews. 
    Please analyze the following review and provide a summary, sentiment (positive, negative, neutral), 
    and any key points mentioned in the review.

    Review: "{review_text}"

    Please respond in JSON format with the following structure:
    {{
        "summary": "A brief summary of the review.",
        "sentiment": "positive/negative/neutral",
        "key_points": ["List of key points mentioned in the review."]
    }}
    """
    return prompt
if __name__ == "__main__":
    #Наш сырой отзыв для анализа
    user_review = """Купил эти наушники вчера. Звук чистый, объёмный за свои деньги топ.
    Однако амбушуры слишком жёсткие, уши начинают болеть через час использования"""

    print("1. Формируем сложный промт...")
    final_prompt = analyze_review_promt(user_review)
    print("2. Отправляем промт в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1) # Низкая температура для соблюдения

    print(f"\nСырой ответ от модели:\n{raw_response}\n")

    print("3. Проверка валидность полученного JSON...")
    try:
        parced_json = json.loads(raw_response.strip())
        print("Успех! Данные успешно преобразованы в python dict:")
        print(f"Тоналпьность: {parsed_json.get('sentiment')}")
        print(f"Плюсы:{parced_json.get('pros')}")
        print(f"Минусы: {parced_json.get('cons')}")
    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула невалидный JSON")