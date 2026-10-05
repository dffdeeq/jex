# Прогон jex в Colab через colab-mcp

[colab-mcp](https://github.com/googlecolab/colab-mcp) — локальный мост между агентом и вкладкой Colab в
вашем браузере. Сервер запускается рядом с агентом и поднимает websocket на `localhost` на случайном порту
с одноразовым токеном. Затем он открывает в браузере
`colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=…&mcpProxyPort=…`, и фронтенд Colab сам
подключается к этому `localhost`. Поэтому агент и браузер должны работать **на одном компьютере**: облачная
сессия Claude Code до вашего браузера не дотянется. Нужна сессия Claude Code на вашей машине.

## Настройка (один раз, на своём компьютере)

```bash
# 1. uv (из него запускается сервер)
pip install uv            # или: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. репозиторий
git clone -b claude/async-model-architecture-duby2o https://github.com/dffdeeq/jex.git
cd jex

# 3. подключить colab-mcp к Claude Code (scope user — для всех проектов)
claude mcp add --scope user colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp

# 4. запустить сессию в папке проекта
claude remote-control     # сессия появится в приложении Claude Code (можно писать с телефона/веба)
# или просто: claude
```

Требования colab-mcp: клиент работает локально и поддерживает `notifications/tools/list_changed`. Claude
Code поддерживает. Браузер должен быть залогинен в Google, вкладку Colab нельзя закрывать, компьютер не
должен засыпать.

## Что сказать локальной сессии

> Подключись к Colab (`open_colab_browser_connection`). Я переключу рантайм на T4 GPU. Дальше по шагам
> выполни ячейки из `notebooks/jex_gpu.ipynb`: клон репозитория (токен в секрете Colab `GITHUB_TOKEN`),
> установка, тесты, `extract.py` (студент Qwen2.5-1.5B, учитель Qwen2.5-7B в 4-bit), `train_head.py` с
> абляциями, `evaluate.py`, `bench_latency.py`, `bench_async.py`. Следи за выводом и чини ошибки. В конце
> занеси таблицы в раздел 7 `docs/RESEARCH.md`, закоммить и запушь в
> `claude/async-model-architecture-duby2o`.

Когда соединение установится, у агента появятся инструменты Colab: создавать, менять и запускать ячейки
и ставить пакеты. Тип рантайма (T4) переключите сами в открывшейся вкладке: Runtime → Change runtime type.

Без MCP всё то же самое делается вручную: откройте `notebooks/jex_gpu.ipynb` в Colab и нажмите Run all.
