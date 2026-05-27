# Датасет — как устроено

## Pre-sample (один объект в json)

Один объект = **одно сгенерированное поле** (без траектории). Потом сами нарежете на семплы и посчитаете эксперта.

Внутри:

- **predicate_space** — размер, стены, действия, правило score, как снимается картинка
- **mission** — текст задачи
- **layout** — стены, цель, старт, seed
- **target** — только `goal_pos` (куда идти)

Картинки не в файлах. Кадр — `reset` + `RGBImgPartialObsWrapper` + `plt.imshow` в replay.

## Поле

1. Сетка + рамка ([туториал MiniGrid](https://minigrid.farama.org/content/create_env_tutorial/)).
2. Агент `(1, 1)`.
3. Случайные стены, связность с агентом сохраняется.
4. BFS → связный кусок.
5. Финиш — случайная клетка из куска.

## Запуск

```powershell
.\.venv\Scripts\python Datasets\create_sample.py
.\.venv\Scripts\python Datasets\replay_sample.py 0
```

```powershell
.\.venv\Scripts\python Datasets\create_sample.py --size 9 --n-walls 6 --seed 3
.\.venv\Scripts\python Datasets\replay_sample.py 0 --no-pygame
```

## Файлы

- `maze_presample.py` — генерация поля
- `create_sample.py` — +1 в `dataset.json`
- `replay_sample.py` — показать стартовый кадр

Старый `dataset.json` с `target.trajectory` — пересоздай через `create_sample.py`.
