# Запуск на GPU Kaggle через CLI

`kagglejobs/run.py` запускает задачи jex на бесплатных GPU Kaggle (2×T4, около 30 GPU-часов в неделю)
и забирает результаты. Браузер для этого не нужен.

## Что сделать один раз

1. **Подтвердить телефон** в аккаунте Kaggle: Settings → Phone verification. Без этого ноутбуки не
   получают GPU и интернет, а нам нужен интернет для pip, Hugging Face и Zenodo.
2. **API-токен.** kaggle.com/settings/api → «Generate New Token». Токен должен попасть в переменную
   окружения `KAGGLE_API_TOKEN`:
   * в облачной сессии Claude Code: в настройках окружения (меню окружения в заголовке сессии → Edit)
     добавить переменную `KAGGLE_API_TOKEN`; новая сессия подхватит её сама. В чат токен не вставлять;
   * локально: `export KAGGLE_API_TOKEN=...` или файл `~/.kaggle/access_token`.
3. `pip install kaggle` (проверено с 2.2.4). Проверка: `kaggle quota` показывает недельную GPU-квоту.

GitHub-токен на Kaggle не нужен. Код уезжает приватным датасетом `<user>/jex-src`, при каждом запуске
выходит его новая версия, собранная из git-отслеживаемых файлов рабочей копии.

## Запуск

```bash
python kagglejobs/run.py smoke       # ~10 мин: GPU, тесты, мини-бенчмарк Jev (3 датасета × 5 запросов)
python kagglejobs/run.py jev-bench   # бенчмарк Jev: Qwen2.5-1.5B, Qwen3.5-4B, Qwen3.5-9B (4-bit), 36 × 30
python kagglejobs/run.py full        # данные → головы → LoRA → eval → бенчмарк Jev (LoRA, 1.5B) → latency/async

python kagglejobs/run.py jev-bench --set bench_limit=100     # переопределить любой ключ конфига
python kagglejobs/run.py full --no-wait                     # отправить и не ждать
python kagglejobs/run.py status full                        # статус последнего запуска
python kagglejobs/run.py output full                        # скачать результаты ещё раз
python kagglejobs/run.py full --dry-run                     # только собрать build/kaggle/jex-full/
```

Что происходит при запуске:
1. Собирается и загружается `jex-src`.
2. В `build/kaggle/jex-<job>/` пишутся `kernel-metadata.json` и `job.py` с внедрённым `CONFIG`
   (машина `NvidiaTeslaT4`, интернет включён, датасет `jex-src` подключён).
3. Выполняется `kaggle kernels push`.
4. Статус опрашивается раз в минуту.
5. `kaggle kernels output` скачивает всё в `runs/kaggle/<job>/`:
   * `results/summary.json`: конфиг, упавшие шаги, время;
   * `results/jev_compare.md`: таблица против Jev, Qwen3.8-27B и Gemma-4-E4B;
   * `results/eval*.json`, `bench_*.json`, `train_*.json`;
   * `lora/`: адаптер из задачи `full`;
   * `jex-<job>.log`: полный лог.

Шаги в `kagglejobs/job.py`: `setup`, `tests`, `pipeline`, `lora`, `jev_bench`, `latency`, `async`. Если шаг
падает, следующие шаги и сохранение результатов всё равно выполняются, а упавший шаг попадает в
`summary.json`. В `bench_models` значение `"@lora"` означает чекпойнт, созданный раньше в той же задаче.

Перед отправкой `job.py` проверялся локально, в симуляции путей Kaggle (`JEX_INPUT_DIR`, `JEX_WORK_DIR`,
`JEX_OUT_DIR`): setup → tests → jev_bench → таблица сравнения.

## Ограничения Kaggle

* Сессия длится не больше 12 часов; `full` укладывается в лимит `timeout` из пресета (11 ч).
* `/kaggle/working` (то, что скачивается) ограничен 20 ГБ. Поэтому признаки и харнесс лежат в
  `/kaggle/tmp`, а в выход попадают только результаты и адаптер.
* Hugging Face скачивается анонимно. При ошибках 429 повторите запуск позже; токен HF можно добавить
  секретом Kaggle через веб-интерфейс ноутбука.
