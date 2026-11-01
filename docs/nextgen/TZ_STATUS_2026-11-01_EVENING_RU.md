# ТЗ r8: что сделано и что осталось (1.11.2026, вечер)

**Где лежит:**

- ветка `MarsWise/openTPU:feature/r7-fpga`, HEAD `63f23383`;
- PR #1 открыт, draft, не слит.

**Сверено с:**

- исходным ТЗ проекта;
- `MASTER_PLAN_R8_RU` (Codex);
- `TZ_ADDENDUM_2026-11-01`;
- `EXPANSION_PLAN_2026-11-01`;
- ревью Codex `60c81131`.

**Уровни доказательств:**

| Уровень | Что значит |
|---|---|
| SRC | только код |
| ISA | симулятор ISA |
| RTL | Verilator, RTL == ISA бит в бит |
| FRAME | эталонный fp32-расчёт на CPU |
| ROUTE | Vivado, разводка |
| PHYS | на плате |

**Статусы:** ✅ сделано, 🟡 частично, ❌ нет, ⏸ отложено.

## 1. Коммиты за сегодня (ветка `feature/r7-fpga`)

| Коммит | Что |
|---|---|
| `23cdfb14` | r8 RTL: VOP RAND, сэмплинг на устройстве, TMM CQ, QST Q4, TFN atan, LDX |
| `4851de3d` | Тесты под «первый токен на карте» |
| `0ad35360`, `60c81131` | Шум Diffusion Policy и SmolVLA на устройстве (`--device-noise`) |
| `456b95ac` | TMM T5 (плотные триты), steering на устройстве, декодер PTQ1_0; 3 исправления по ревью Codex |
| `2d5e98b8`, `f95aac92` | Bonsai 2 из GGUF: `GGUFWeights`, `spec_from_gguf`; 4B/9B — явный отказ |
| `2426e5b2`, `05c872f2` | Скрипт скачивания моделей: 2 МБ/с, докачка, SHA-256; заметки по Bonsai 2 GGUF |
| `63f23383` | WIP C3: VOP BFLY (Адамар) в ISA, симуляторе, ядре и RTL; деквантизация K-квантов GGUF |

## 2. RTL и сборки

| № | Пункт | Статус | Доказательство |
|---|---|---|---|
| A1 | V_RAND, сэмплинг, шум, штраф за повтор на карте | ✅ | RTL 4/4, ISA |
| A1 | Генерация RTL S=2 (split) | ❌ | В контейнере не хватает памяти, передано Codex |
| A2 | TMM CQ (CQ1–CQ4, LUTSET) | ✅ | RTL 5/5; регрессия TMM 39/39 |
| A3 | Sign-off GRQ (CDC, TIMING-54, I/O) | ❌ | Нужен Vivado — сторона пользователя |
| B2 | Сборка r8 m8 133 МГц | 🟡 | Скрипты готовы, ошибка `$r8` исправлена; запуск — пользователь |
| C1 | QST Q4 | ✅ | RTL 6/6 |
| C1 | 6-битный V | ⏸ | |
| C2 | PAIR/QUAD + GSC | ⏸ | После замера GRQ на плате |
| **C3** | **Блок Адамара (VOP BFLY, CAPS2 bit16)** | ✅ RTL | RTL == ISA 1/1, ISA 3/3; ×2,0 на преобразовании; синтез — нет |
| C6 | PCIe MSI / litepcie | ⏸ | Отложено решением пользователя |
| C8 | atan в TFN | ✅ | RTL 2/2 |
| C9 | int16-активации | ⏸ | Только если не хватит hi+lo |
| C10 | TMM T5 — плотные триты, 1,6 бита на вес (CAPS2 bit15) | ✅ | RTL 2/2, ISA 8/8; регрессия TMM 39 passed |
| C11 | LDX + билинейная выборка | ✅ | RTL 5/5 + 1/1; LDX ~9× быстрее цикла RLD+LD |
| R3 | Fused affine (LN/RMSNorm) | ❌ | Следующий после C3 |
| R4, R6, R7 | MAC, QST T4, m8 133 МГц | ✅ | RTL / ROUTE |
| — | Steering на устройстве | ✅ | ISA 5/5 (не требовался, сделан) |

**C3 подробно.** Новая функция VPU — `BFLY`: один шаг бабочки за проход по всей строке. Партнёр берётся так:

- при шаге s ≥ 8 — через порт B по адресу `ic xor s`;
- при s < 8 — перестановкой данных внутри чанка.

Результаты:

- ISA: бит-в-бит равно прежнему преобразованию на strided-VOP; на 9 VOP меньше (3/3).
- RTL: RTL == ISA бит-в-бит (1/1, Verilator, шаги и меньше, и больше 8 дорожек).
- Замер RTL (m8, ширина 2048, один ряд): VOP-часть преобразования 5931 → **2980 тактов (×1,99)**;
  всё ядро с LD/ST 6778 → 3827 тактов (×1,77). Каждый шаг теперь ~271 такт (2048/8 + накладные).
- Не сделано: синтез, ресурсы и тайминг (добавлено мультиплексирование дорожек и XOR в адресе
  порта B; ждёт сборки r8 с `VBFLY=1`).

## 3. FPGA-only (хост не считает)

| Пункт | Статус |
|---|---|
| Первый токен ответа, uniform, шум, штраф за повтор на карте | ✅ ISA |
| Strict FPGA-only с сэмплингом при VRAND | ✅ исправлено по ревью Codex |
| Шум Diffusion / SmolVLA на карте | ✅ ISA (тесты пропускаются без lerobot) |
| Первый токен MTP-префилла | ❌ |
| Позиции до `conv_k − 1` в LFM2 | ❌ |
| Аудит «вычислений на хосте = 0», негативные тесты | ❌ |
| Исполнитель графа на карте (YOLOX, Diffusion, SmolVLA) | ❌ |
| Планировщик памяти под полные модели | ❌ |
| MoE полностью без хоста | ❌ |

## 4. Модели

### Что есть на диске `C:\ai-models`

| Модель | Файлы на диске | Статус |
|---|---|---|
| Bonsai 2 2B | GGUF PTQ1_0 / PQ2_0 / Q2_0 | ✅ PTQ1_0 == PQ2_0 на всех 151 тензоре (SW). ✅ FRAME: «The capital of France is» → « Paris». ❌ ISA/RTL через T5. ❌ плата |
| Bonsai 2 4B / 9B | GGUF PTQ1_0 / PQ2_0 (+MTP, DFlash, mmproj) | ❌ Заблокировано: 32 V-головы на 16 K. Ни один из 4 вариантов порядка голов не даёт «Paris». Нужен эталон — см. ниже |
| MiniCPM5 2B | только GGUF Q4_K_M | ❌ Деквантизатор K-квантов написан, не проверен на модели |
| Granite 4.2 3B / 8B | только GGUF Q4_K_M | ❌ То же |
| Dory 4B | только GGUF Q5_K_M | ❌ То же |
| PaddleOCR-VL 1.6 | safetensors | ❌ (LDX + билинейная выборка готовы) |
| MusicGen small | safetensors | ❌ |
| Kokoro 82M | `.pth` | ❌ (atan2, шум готовы) |
| Ouro 1.4B / 2.6B | safetensors | 🟡 конвертер и MLP (Codex) |
| SmolVLA base | есть | 🟡 ISA resident; RTL/плата — нет |
| Cactus needle | есть | 🟡 TMM CQ готов, реальный чекпойнт не подключён |

Как разблокировать 4B/9B. Подключить Qwen3.5-4B в формате llama.cpp (Huihui Q4_K_M на диске) и найти по ней порядок голов. Прогон был остановлен пользователем. Второй путь — скачать MLX-версию: `download_models.cmd bonsai2-4b-mlx`.

### Чего на диске нет

| Модель | Что есть вместо | Как получить |
|---|---|---|
| LFM3 1.7B | — | `download_models.cmd lfm3` |
| SD 1.5 | SDXL и FLUX.2-klein | `download_models.cmd sd15` |
| FunctionGemma 270M | только README, кэш пустой — веса не скачались (gated) | `huggingface-cli login`, затем скачать |
| ReasonLite 0.6B | — | `download_models.cmd reasonlite` |
| Safetensors MiniCPM5 / Granite / Dory | только GGUF | `download_models.cmd minicpm5 granite dory` |

### Что работает на плате (r5, greedy)

Qwen3 / 3.5, LFM2, BitNet, Bonsai 4B (старый, MLX), MTP k=1, vLLM greedy.

### Только ISA, на плате не проверено

- YOLOX-Nano (FPGA-only на ISA);
- MiniCPM-V: ответ по картинке на плате не проверен;
- vLLM: сэмплинг и logprobs.

### Не начато

FFN pruning, speculative decoding k > 1.

## 5. Проверки

| Набор | Результат |
|---|---|
| TMM RTL после T5 | 39 passed |
| ISA / unit (T5, PTQ1, хост, наблюдаемость, таргеты, placement, CQ, ISA) | 163 passed; 1 тайминговый флейк, отдельно проходит |
| Полный набор, часть 0 из 7 | 412 passed, 33 skipped, 1 failed (`test_compile_worker_exits_with_its_parent`, не разобран) |
| Полный набор, части 1–6 | Не запускались |
| BFLY | ISA 3/3; RTL 1/1 |

## 6. Что осталось — по порядку

**Я:**

1. C3: включить VBFLY в сборку r8 (скрипты), синтез — у пользователя.
2. R3: fused affine (LN/RMSNorm).
3. Bonsai 4B/9B: порядок голов по Qwen3.5-4B (Huihui) или по MLX-версии.
4. Bonsai 2B на ISA и RTL через T5.
5. MiniCPM5 / Granite / Dory из GGUF K-квантов: проверка FRAME, затем ISA.
6. FPGA-only: MTP-префилл, LFM2 позиции, аудит, исполнитель графа, планировщик памяти.
7. PaddleOCR-VL, Kokoro, MusicGen, LFM3, SD 1.5 (последние два — после скачивания).
8. Полный набор тестов, части 1–6.

**Пользователь:**

1. Сборка r8 в Vivado, затем загрузка на плату.
2. Sign-off GRQ.
3. Скачивание: `bonsai2-4b-mlx lfm3 sd15 reasonlite`, FunctionGemma — после логина в HF.

**Codex:** регрессия S=2, перенос r8 в свою копию. Свои незакоммиченные `board.py` и `chat.py` Codex сохраняет.

**Отложено:** C6 PCIe/MSI, C1 6-битный V, C2 PAIR/QUAD+GSC, C9.
