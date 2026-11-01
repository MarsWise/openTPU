# ТЗ r8: что сделано и что нет (1.11.2026, вечер)

Сверка с действующим ТЗ:

- `docs/nextgen/r8/MASTER_PLAN_R8_RU.md` (Codex);
- `EXPANSION_PLAN_2026-11-01_RU.md`;
- `r8/TZ_ADDENDUM_2026-11-01_RU.md`;
- исходное ТЗ проекта.

Источники фактов:

- код `feature/r7-fpga` = `b9fe0ca6`;
- рабочая копия Codex в `C:\myOpenTPU\openTPU-r7` (не закоммичена) и его `r8/PROGRESS_2026-11-01.md`;
- `VERIFIED_RESULTS.csv`;
- мои изменения этого вечера (первый токен на карте, ниже).

**Статусы:**

| Статус | Значение |
|---|---|
| ✅ | сделано на уровне, указанном в скобках |
| 🟡 | частично |
| ❌ | нет |
| PLAN | только анализ или план |

Уровни: ISA, RTL, SYNTH, ROUTE, PHYS (на плате), FPGA-ONLY PHYS.

## Короткий итог

- **Сделано и проверено на плате:**
  - LLM-декод (Qwen3, Qwen3.5, LFM2, BitNet, Bonsai 4B, MTP);
  - vLLM greedy;
  - BitNet: батч через vLLM.

  Всё это на r5. Ни одного физического прогона r7 пока нет: прошивка GRQ собрана, но не загружена.
- **Сделано в коде, проверено в ISA/RTL, без платы:**
  - YOLOX FPGA-only;
  - Diffusion Policy и SmolVLA resident;
  - препроцессинг MiniCPM-V на карте;
  - сэмплинг и logprobs vLLM;
  - GRQ;
  - заготовки Cactus и Ouro.
- **Не начато (только план):**
  - исполнитель графа на карте, планировщик памяти, единый аудит;
  - `V_RAND`, TMM CQ;
  - LFM3, MiniCPM5, Granite, Dory;
  - Kokoro, MusicGen, SD 1.5;
  - steering, ReasonLite, FunctionGemma, PaddleOCR-VL, Bonsai 9B;
  - KV 4/6, FFN pruning, speculative k > 1.

---

## Этап 0 — основа и воспроизводимость

| Пункт ТЗ | Статус | Что есть / чего нет |
|---|---|---|
| SHA, manifest, сохранение r5 | ✅ | r5 `45b1c1d3`; r7 `b9fe0ca6` (manifest, bitstream `7ecc66e3…`, `CFG_HASH d9b863d5`) |
| Ревью скриптов: коды выхода, lock, логи, ECC до и после | 🟡 | ECC guard и `pipefail` у Codex (не закоммичены); мои `r7_*.cmd`/`.sh` в ветке. Слить и закоммитить |
| Расширенные тесты GRQ | 🟡 | базовые RTL==ISA плюс тесты Codex (`test_tmm_grq.py`, +32 строки). Не покрыты: нулевые и разнознаковые масштабы, PAIR/QUAD с GSC, reset, конкуренция DMA |
| Полная регрессия r5/r6/r7 | ❌ | выборочно: GRQ/GSC/Bonsai/Cactus/Ouro — 61 и 44 passed. Полный прогон после 31.10 не делался (`r7_sim_tests.cmd`) |
| Sign-off сборки r7 (CDC, `TIMING-54`, I/O, bus skew, финальный DCP) | ❌ | WNS +0,003 / WHS +0,004 нс, WPWS 0,000; отчёты для следующей сборки подготовлены в `build.tcl` (Codex) |

## Этап 1 — Bonsai и GRQ

| Пункт | Статус | Детали |
|---|---|---|
| Baseline r5 | ✅ PHYS | 0,68 ток/с; cosine logits 0,999926; 17,9 ГБ чтений на токен |
| GRQ (каждый чанк читается один раз) | ✅ RTL, ✅ ROUTE | не загружен; ожидание до ~6,8× на GEMV (оценка) |
| Сравнение GRQ 0/1 на плате | ❌ | скрипт `bonsai_profile_grq_compare.sh` есть; нужна ваша загрузка прошивки |
| **Первый токен на карте** | ✅ код (сегодня) | см. «Изменение этого вечера»; на плате не проверено |
| GSC reuse с PAIR/QUAD | ❌ | после замера GRQ |

## Этап 2 — общий FPGA-only runtime и память

| Пункт | Статус | Детали |
|---|---|---|
| Графы и operator coverage по точным чекпойнтам | 🟡 | матрицы r7 есть (`R7_OPERATOR_COVERAGE`), по новым моделям нет |
| Планировщик памяти на полные модели | ❌ | `nextgen/planner.py` — консервативный аллокатор и аудит; к resident и LLM не подключён |
| Исполнитель графа на устройстве | 🟡 | `HALT CHAIN` работает для LLM-декода; resident-графы (YOLOX, Diffusion, SmolVLA) запускает хост по одной программе |
| Единый strict FPGA-only и рантайм-аудит | 🟡 | `--strict-fpga`, `--strict-fpga-only` (сегодня работает для greedy-чата), `OTPU_VLLM_FPGA_ONLY`. Нет общего аудита «host numeric ops = 0» и негативных тестов с перехватом numpy |
| RNG на устройстве (`V_RAND`) | ❌ | Philox в `rtl/exp/philox` (RTL testbench), к ISA не подключён; Box–Muller нет |
| Общие ядра (flash-attention, LSTM, iSTFT, convT2d, CFG, сэмплер K ≥ 250, L2-norm…) | 🟡 | есть attention с масками, conv1d/2d, convT1d, GroupNorm, сэмплер K ≤ 64, resize/letterbox; остальное нет |

## Этап 3 — сценарии r7

| Модель | Статус | Детали |
|---|---|---|
| YOLOX-Nano | ✅ ISA FPGA-only | letterbox и NMS на карте; плата, mAP и латентность — нет (нужна прошивка с VU8, r7) |
| Diffusion Policy PushT | 🟡 ISA resident | шум и нормализация на хосте, шаги запускает хост |
| SmolVLA | 🟡 ISA resident (tiny) | base не прогнан; шум и предобработка на хосте; Euler-цикл не цепочкой |
| MiniCPM-V | 🟡 | препроцессинг на карте (ISA/RTL, `8a4a7ac3`), башня и LM — upstream; сквозной ответ по картинке на плате — нет; `rope_index` на хосте |
| BitNet | ✅ PHYS (greedy) | 17,5 ток/с (int8 head), 20,9 (fp4 head); первый токен на карте — сегодня, в коде. Сэмплинг на плате — нет |
| vLLM | 🟡 | greedy PHYS (до 28,9 ток/с при 8 слотах, с ECC-ошибками); сэмплинг и logprobs — ISA; min-p, penalties, logit_bias, prompt_logprobs явно отклоняются; многослотовая отмена на плате не проверена |

## Этап 4 — Cactus CQ

| Пункт | Статус |
|---|---|
| Парсер CACT-84 и эталон CQ2/CQ4 | ✅ код (Codex) |
| Преобразование активаций на VPU | ✅ ISA, RTL==ISA (синтетика) |
| Офлайн CQ → INT8 baseline | ✅ ISA, RTL==ISA (синтетика) |
| Автономный CQ-декодер | ✅ OOC SYNTH: 10 437 LUT параллельный, 1 754 LUT потоковый; не встроен |
| **TMM CQ в MXU** (ISA, компилятор, CAPS) | ❌ |
| Реальные CQ-чекпойнты, полная модель | ❌ |

## Этап 5 — Ouro-2.6B-Thinking

| Пункт | Статус |
|---|---|
| Чекпойнт, конфиг, хэш | ✅ |
| Конвертер INT4 v2 (выравнивание 128, strides, SHA, 533 тензора, `--verify-only`) | ✅ (Codex) |
| Блок MLP (RMSNorm, SwiGLU, residual) | ✅ ISA, RTL==ISA (синтетика) |
| Attention, рекуррентный KV `step*48+layer`, 1–4 прохода, early exit, исполнитель | ❌ |
| Сверка с PyTorch по проходам, плата | ❌ |

## Этап 6 — новые LLM

| Модель | Статус | Главное, чего нет |
|---|---|---|
| LFM3-1.7B | PLAN | conv k=7, SWA-256, partial RoPE, head gates, sandwich norms (без нового RTL) |
| MiniCPM5-2B | PLAN | m4-сборка, GQA 8:1 при MCOLS 4, контекст 2–8K |
| Granite 4.2 3B | PLAN | m4, GQA 5:1 → 4+1, head_dim 64 при D=128, множители |
| Dory 1.0 4B | PLAN | весь граф nGPT (L2 justnorm, нормированный residual, рекуррентность) |

## Этап 7 — речь, музыка, изображения

| Модель | Статус |
|---|---|
| Kokoro-82M | PLAN (BiLSTM, AdaIN, iSTFT, atan2, шум — нет) |
| MusicGen-small | PLAN (T5 relative bias, CFG, top-k 250 ×4, EnCodec — нет) |
| SD 1.5 | PLAN (flash-attention 1–4K, CFG, VAE, шум — нет) |

## Дополнение к ТЗ (N1–N9)

| № | Направление | Статус | Что нужно |
|---|---|---|---|
| N1 | Activation steering | PLAN | ядро на RDOT/MUL/ADD, профили в DDR, API, ключ KV; RTL не нужен |
| N2 | ReasonLite-0.6B | 🟡 | архитектура = Qwen3 (PHYS); нужны загрузчик, шаблон и сверка |
| N2 | FunctionGemma-270M | PLAN | граф Gemma 3, голова 262K, tool-call парсер |
| N2 | PaddleOCR-VL-0.9B | PLAN | NaViT позиционки, проектор, ERNIE-4.5; layout RT-DETR — уровень 2 |
| N2 | Bonsai 2 9B | PLAN | парсер GGUF PTQ1_0/PQ2_0/Q2_0, перепаковка в 2-битные слоты, память 4 GiB |
| N3 | Нативный VLA | 🟡 | SmolVLA resident (ISA); нужны цепочка Euler, `V_RAND`, буфер action chunks, асинхронный рантайм |
| N6 | FFN pruning | PLAN | офлайн-прунер, разная ширина FFN по слоям в Spec |
| N7 | KV 4/8 бит | 🟡 | 8 бит есть; 4/6 бит — RTL (`QST` 4/6 + int6), проверка качества по dKL |
| N8 | Speculative | 🟡 | MTP k=1 PHYS (Qwen3.5); нет k > 1, драфт-модели, rejection sampling на карте |
| N9 | Sparse MoE | 🟡 | роутер, top-k и эксперты на карте (upstream); полностью резидентный режим без хоста — нет |

## Сборки

| Сборка | Статус |
|---|---|
| r7 GRQ m8 `b9fe0ca6` | ROUTE, sign-off не закончен, не загружена |
| r8 m8 (+`V_RAND`, +TMM CQ, +`QST` 4/6) | не начата |
| r8 m4 (MiniCPM5, Granite) | не начата |
| универсальная r8 | не начата; LUT 67,6 %, DSP 74,8 % уже заняты |

---

## Изменение этого вечера: первый токен на карте

- **`otpu-chat`** (все LLM: Qwen3, Qwen3.5, LFM2, Gemma 4, BitNet, Bonsai, MiniCPM-V):
  - prefill без последнего токена промпта;
  - последний токен подаёт цикл на карте (`generate_card`), его LM head выбирает первый токен ответа;
  - логиты на хост не уходят (как уже было в `otpu-serve` и vLLM `fpga_only`);
  - снапшот для отката — не позже предпоследнего токена.
  - Вариант по-старому: `--host-first-token`.
- **`--strict-fpga-only` теперь работает** для greedy без `--mtp`. При сэмплинге отказывает: uniforms пока с хоста, нужен `V_RAND`. При MTP тоже отказывает: первый токен MTP-префилла пока выбирает хост. Любой ход, где первый токен выбрал бы хост, завершается ошибкой.
- **`placement.py`**: новые стадии «first token of a reply», «sampling uniforms», «repetition penalty of the prompt» видны в отчёте `--strict-fpga`.
- **`tools/bitnet_engine.py --mode generate`**: первый токен на карте (кроме `--save` и `--host-first-token`).
- **Осталось хостом:**
  - первый токен MTP-префилла (чат и сервер);
  - позиции до `conv_k − 1`;
  - vLLM без `fpga_only`.

## Что делать дальше (по ТЗ, по порядку)

1. **Этап 0:**
   - коммит изменений Codex (ECC guard, скрипты, Cactus, Ouro) — его рабочая копия меняет `chat.py`, перед `git pull` ему надо закоммитить;
   - полная регрессия `r7_sim_tests.cmd`;
   - разбор sign-off.
2. **Этап 1:** загрузка GRQ (ваше разрешение) → замер Bonsai GRQ 0/1 и проверка первого токена на карте.
3. **Этап 2:**
   - `V_RAND` (Philox → VOP), затем сэмплинг и шум на карте;
   - исполнитель графа на `HALT CHAIN` для resident-графов;
   - планировщик памяти;
   - единый аудит.
4. Первый токен в MTP-префилле на карте.
5. **Этапы 4–5:** TMM CQ, исполнитель Ouro.
6. **Этапы 6–7 и N1–N9** — по порядку из ТЗ.
