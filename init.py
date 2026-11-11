import fire
from llama_cpp import Llama
import pynvml  # Библиотека для мониторинга NVIDIA
import time  # Импортируем время, мудила

from torch.library import define

# Путь к модели
SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
model_path = "./YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"
# model_path = "./saiga_yandexgpt_8b.Q4_K_M.gguf"
# model_path = "./saiga_gemma3_12b.Q8_0.gguf"
# model_path = "./model-q8_0.gguf"
start_time = time.perf_counter()  # Засекаем время

try:
    # Пытаемся получить количество доступных GPU
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[INFO] Найдено GPU: {device_count}")

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU #{i}: {name}, Памяти: {mem_info.total // 1024 ** 2} MB")

    pynvml.nvmlShutdown()
except Exception as e:
    print(f"[WARNING] Не удалось получить информацию о GPU: {e}")

# Инициализируем модель и ВСЕ параметры указываем тут
llm = Llama(
    model_path=model_path,
    n_gpu_layers=20,  # Number of layers to load on GPU
    n_ctx=4096,  # Context length
    use_mlock=False,  # Optional: lock memory for performance
    verbose=False,  # Print layer-device map
)


def is_defined(var_name):
    """Проверяет, определена ли переменная"""
    return var_name in globals() or var_name in locals()


messages = [{"role": "system", "content": SYSTEM_PROMPT}]
while True:
    end_time = time.perf_counter()  # Конец замера
    elapsed = end_time - start_time
    print(f"### ОТВЕТ ЗА {elapsed:.2f} СЕКУНД ###")
    # Дополнительно: считаем скорость генерации

    user_message = input("User: ")

    start_time = time.perf_counter()  # Засекаем время

    messages.append({"role": "user", "content": user_message})
    assistant_message = ""
    start_time = time.perf_counter()  # Засекаем время
    for part in llm.create_chat_completion(
        messages,
        temperature=0.1,
        repeat_penalty=1.1,
        stream=True,
    ):
        delta = part["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
            assistant_message += delta["content"]
    messages.append({"role": "assistant", "content": assistant_message})
