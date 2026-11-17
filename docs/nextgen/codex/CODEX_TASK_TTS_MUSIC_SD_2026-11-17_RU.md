# Задание для Codex: Kokoro-82M, MusicGen-small, Stable Diffusion 1.5 на openTPU r8 (FPGA-only)

Дата: 17.11.2026. Заказчик: MarsWise/openTPU, ветка `feature/r7-fpga` (r8). Это задание на
**три независимые части**; каждую можно делать и сдавать отдельно. Интеграцию, сборку прошивки
и прогоны на плате делаем мы (Claude) — от тебя код, тесты на симуляторе ISA и отчёт.

## 0. Общие правила (обязательно)

### 0.1 Git и публикация
- Работай в **своём клоне**: `git clone https://github.com/MarsWise/openTPU.git C:\myOpenTPU\codex-gen`
  (или в WSL), ветка `feature/r7-fpga` от коммита не старше `27f90e59`. Локальная ветка —
  `codex/<модель>`.
- **Ничего не пушь.** Ни в MarsWise, ни тем более в FeSens. Сдача — `git format-patch` от базового
  коммита в `C:\myOpenTPU\_claude\codex\<модель>\` плюс `REPORT_<модель>.md`.
- Не трогай `C:\myOpenTPU\openTPU-r8` (это чистая копия для сборок Vivado), `C:\myOpenTPU\_claude\r8\`
  (скрипты платы) и `C:\myOpenTPU\openTPU.worktrees\...`.

### 0.2 Железо — не трогать
- **Никаких** запусков Vivado, загрузки прошивки, JTAG, SSH на хост платы (192.168.0.157),
  обращений к `/dev/xdma*`. Плату используем мы, параллельные прогоны её ломают.
- **RTL (`rtl/`) не менять.** Если какую-то операцию действительно нельзя выразить
  существующими инструкциями — опиши её в отчёте (что, почему, сколько тактов/ресурсов
  сэкономит) и, если хочешь, положи прототип с тестбенчем в `rtl/exp/<имя>/`. В рабочий RTL не встраивать.

### 0.3 Модели и данные
- Скачивание весов **не быстрее 2 МБ/с** (`--limit-rate 2M` / `HF_HUB_DOWNLOAD_TIMEOUT` и т.п.).
  Только `safetensors` / `.pth` через `torch.load(weights_only=True)`. **Не запускай код из папок
  моделей** (`modeling_*.py`, `trust_remote_code`, pickle): эталон — через пакеты `transformers`,
  `diffusers`, `kokoro` из PyPI в изолированном venv.
- Запиши в отчёт точный repo id, ревизию (commit sha на HF) и sha256 каждого файла весов.

### 0.4 Что значит «FPGA-only» в этом проекте
- На устройстве: всё, что после токенизации/текстового препроцессинга и до готового тензора
  выхода (аудио-сэмплы, пиксели). Хост имеет право: токенизатор / G2P (фонемизация), чтение и
  запись файлов, запись входа в DRAM и чтение выхода, запуск программ.
- В `--strict-fpga` любая неподдержанная операция — **явная ошибка**, а не тихий CPU-фолбэк.
- Каждая часть должна печатать таблицу размещения (`opentpu/llm/placement.py`-стиль): что где
  выполнилось.

### 0.5 Целевая конфигурация
- r8 m8: D = 128, MCOLS = 8, LANES = 8, 133,33 МГц, DRAM 2 × 2 ГиБ (DDR3-1066, пик 17,1 ГБ/с).
  `opentpu.isasim.board_config()` + возможности r8 (CAPS2: TGSC, TBIN, VCMP, VU8, TGRQ, VRAND,
  TCQ, TATAN, QST4, LDX, TD5, VBFLY, VAFF, QST6, VCSWP, TGQ4) — смотри `targets/ypcb-00338.toml`,
  `opentpu/host/regs.py` (`CAP2_*`), `docs/isa.md`.
- Форматы весов: int8, int4, fp4, nvfp4, троичные (TMM), смешанные (`opentpu/llm/formats.py`).

### 0.6 На что опираться (не изобретать заново)
- Резидентный граф целиком на устройстве: `opentpu/resident.py` (`Graph`, `dense_k`,
  `group_norm_k`, `group_norm2d_k`, `attention_k`, `film_k`, `binary_k`, `unary_k`, `run_isa`),
  примеры: `opentpu/smolvla_resident.py`, `opentpu/diffusion_resident.py`.
- Ядра: `opentpu/kernels/` — `conv.py` (conv2d через MXU, depthwise, maxpool, upsample2x),
  `lstm.py`, `ddpm.py`, `randn.py` (нормальный шум на устройстве), `resize.py`, `sdpa.py`,
  `attention.py`, `linear.py`, `eltwise.py`, `hadamard.py` (BFLY), `sort.py`, `vit.py`.
- Язык ядер: `opentpu/language.py` (`ol.jit`, `ol.load/store`, VOP-функции), компилятор
  `opentpu/compiler.py`, симулятор `opentpu/isasim.py`.
- Функции VPU (TFN): sigmoid, SiLU, tanh, GELU, softplus, Mish, sin, cos, atan, exp2, log2,
  recip, rsqrt (`opentpu/tfn.py`, `docs/isa.md`).
- Тесты-образцы: `tests/test_conv.py`, `tests/test_lstm.py`, `tests/test_resident.py`,
  `tests/test_conv1d_diffusion.py`, `tests/test_ddpm_step.py`.

### 0.7 Проверка и отчёт (на каждую часть)
1. **Операторы по одному**: каждая новая операция/ядро — pytest против эталона numpy/PyTorch
   fp32 на симуляторе ISA, включая крайние размеры (нечётные длины, не кратные 128 каналы).
2. **Слои/блоки**: промежуточные тензоры против эталонного фреймворка (cos ≥ 0,999 для int8,
   порог для 4 бит — обосновать).
3. **Вся модель на ISA** на маленьком входе: метрика ниже по каждой модели.
4. **Покрытие**: таблица операторов модели → чем выполнено (MXU / VPU / потоковый блок / хост —
   хостом только разрешённое из 0.4).
5. **Оценка производительности на r8**: такты из ISA-симулятора (`cycles`), байты DRAM на шаг,
   ожидаемое время на плате; помечать как ОЦЕНКА (ISA SIMULATED), не как измерение.
6. `pytest -q` на изменённых/новых тестах и `tests/test_resident.py tests/test_conv.py
   tests/test_lstm.py` — без регрессий. Перечисли все пропуски и падения.
7. Классы данных — как в `docs/nextgen/VERIFIED_RESULTS.csv` (SOURCE INSPECTION, SOFTWARE
   VERIFIED, ISA SIMULATED, …). Не выдавать симуляцию за плату.

Сдача: патчи + `REPORT_<модель>.md` (что сделано, файлы, команды запуска тестов, результаты,
что не получилось, нужны ли изменения RTL) + готовая строка(и) для `VERIFIED_RESULTS.csv`.

---

## 1. Kokoro-82M (TTS, текст → речь 24 кГц)

- **Модель:** `hexgrad/Kokoro-82M` (Apache-2.0), v1.0; голос — один из voicepack’ов (например
  `af_heart`), стиль-вектор голоса — вход (256-мерный), хранится на устройстве.
- **Архитектура (StyleTTS2 + iSTFTNet, без диффузии):** PL-BERT/ALBERT текстовый энкодер →
  предиктор длительностей/F0/энергии (BiLSTM, AdaIN) → выравнивание по длительностям → декодер
  iSTFTNet: conv1d, conv-transpose (апсемплинг), AdaIN-ResBlocks с Snake-активацией
  (`x + sin²(αx)/α`), генератор гармоник (SineGen: sin фаз), iSTFT с малым n_fft.
- **Хост может:** G2P (`misaki`/espeak) → фонемы → id. Всё остальное — на устройстве.
- **Ожидаемые решения без нового RTL:**
  - Snake: `sin` есть в TFN, остальное — VOP MUL/ADD.
  - conv-transpose 1d: как conv1d по вставленным нулям или разложение по фазам шага
    (sub-pixel) через `kernels/conv.py`.
  - iSTFT при малом n_fft (параметры — из `config.json`: `gen_istft_n_fft`, `gen_istft_hop_size`; проверить): обратное ДПФ как матричное умножение на MXU
    (матрица cos/sin) + overlap-add; BFLY не обязателен.
  - BiLSTM: `kernels/lstm.py` дважды (прямой/обратный проход).
  - Выравнивание по длительностям: целочисленные повторы строк через `LDX` (индексы считает
    устройство из предсказанных длительностей; округление — как в эталоне).
- **Критерии:** мел/выход декодера и волна против `kokoro` (PyPI) на фиксированной фразе
  (≥ 3 с): cos волны ≥ 0,99 и SNR ≥ 20 дБ для int8 (обосновать для 4 бит); длительности —
  совпадение целых. WAV-файлы эталона и устройства (ISA) — в отчёт.
- **Оценка:** такты на секунду речи (цель — быстрее реального времени на r8).

## 2. MusicGen-small (текст → музыка 32 кГц)

- **Модель:** `facebook/musicgen-small` (~300M; **веса CC-BY-NC 4.0** — только некоммерческая
  проверка, отметить в отчёте).
- **Архитектура:** текстовый энкодер T5-base (замороженный) → авторегрессионный трансформер-
  декодер (24 слоя, 1024, cross-attention на T5, 4 кодбука EnCodec с «задержкой», 50 шагов/с,
  classifier-free guidance — два прохода cond/uncond) → декодер EnCodec 32 кГц (conv1d,
  conv-transpose 1d со страйдами 8·5·4·4, 2-слойный LSTM, ELU).
- **Хост может:** токенизатор T5. Всё остальное, включая сэмплинг токенов (top-k = 250,
  температура 1: `VRAND` + `VCSWP`/ARGMAX — см. `opentpu/llm/generate.py`), — на устройстве.
- **Ожидаемые решения без нового RTL:** T5 relative position bias — таблица в DRAM + сложение;
  ELU — `exp2`/`SEL`; conv-transpose — как в Kokoro; LSTM — `kernels/lstm.py`; декодер-
  трансформер — по образцу существующих LLM (`opentpu/llm/llama.py`, `qwen3.py`), KV-кеш.
- **Критерии:** (а) при жадном выборе и фиксированном seed — последовательность кодов против
  `transformers.MusicgenForConditionalGeneration` (совпадение ≥ 95 % токенов на 2 с, логиты
  cos ≥ 0,999 int8); (б) декодер EnCodec на одинаковых кодах: волна cos ≥ 0,99; (в) сэмплинг на
  устройстве — воспроизводимость по seed.
- **Оценка:** такты на шаг (4 кодбука, CFG ×2) и на секунду музыки; байты весов на шаг.

## 3. Stable Diffusion 1.5 (текст → изображение)

- **Модель:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (CreativeML OpenRAIL-M), fp16
  safetensors; эталон — `diffusers.StableDiffusionPipeline` на CPU fp32.
- **Архитектура:** CLIP ViT-L/14 текстовый энкодер (12 слоёв, 77 токенов) → UNet (~860M:
  conv3×3, GroupNorm(32) + SiLU, ResBlocks с time-embedding, self/cross-attention
  трансформер-блоки, down/up-sampling, skip-конкатенации) × N шагов планировщика
  (DDIM или Euler, classifier-free guidance) → VAE-декодер (~49M, conv, GroupNorm, attention
  на 64×64, апсемплинг).
- **Сначала 256×256** (латент 32×32) и 20 шагов; 512×512 — вторым этапом.
- **Хост может:** токенизатор CLIP. Шум — `kernels/randn.py` на устройстве; планировщик — VPU
  (как `kernels/ddpm.py`).
- **Ожидаемые решения без нового RTL:** GroupNorm — `resident.group_norm2d_k`; attention —
  `resident.attention_k`/`kernels/sdpa.py`; conv3×3 — `kernels/conv.py`; nearest-upsample —
  `upsample2x`; веса UNet — 4 бита (fp4/nvfp4), ~0,45 ГБ; активации в DRAM.
- **Память:** посчитай пиковое использование DRAM (веса + активации + скипы) — должно
  уложиться в 4 ГиБ с запасом.
- **Критерии:** (а) CLIP-эмбеддинги cos ≥ 0,999; (б) один шаг UNet на фиксированном латенте:
  cos шума ≥ 0,995 (int8) / обосновать для 4 бит; (в) VAE-декодер на фиксированном латенте:
  PSNR ≥ 30 дБ; (г) вся цепочка, фиксированный seed и промпт: PSNR против diffusers и
  картинка в отчёт. На ISA допускается проверять UNet по блокам, если целиком слишком долго.
- **Оценка:** такты на шаг UNet при 256 и 512, ожидаемые секунды на картинку на r8.

---

## 4. Порядок и сроки (ориентир)

1. Kokoro — 1–2 недели (самая маленькая, лучший первый кандидат).
2. MusicGen — 2–3 недели.
3. SD 1.5 — 3–4 недели (256×256 → 512×512).

После каждой части — сдача по 0.7. Мы применяем патчи на свежий HEAD, гоняем полную регрессию,
собираем (если нужно) и проверяем на плате; результат пишем в `VERIFIED_RESULTS.csv` с
пометкой авторства.
