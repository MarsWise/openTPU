# openTPU r8: сводное состояние по всем ТЗ (Claude + Codex), 1.11.2026

Сводка заменяет отдельные статусы Claude и Codex за 1.11.

**Источники:**

- ветка `MarsWise/openTPU:feature/r7-fpga` (HEAD `a60ff6f2`);
- сводная копия Codex `C:\myOpenTPU\reports\claude-r8-review-a60ff6f2` — не закоммичена, 43 файла (manifest V2);
- копия `C:\myOpenTPU\openTPU-r7` (HEAD `b9fe0ca6`, с незакоммиченными изменениями);
- документы Codex в `docs/nextgen/r8/`: HANDOFF_FULL, INTEGRATION, BONSAI_HEADS_C3_R3, два ревью;
- отчёт Claude TZ_STATUS evening.

**Учтённые ТЗ:**

- исходное ТЗ проекта;
- MASTER_PLAN_R8;
- TZ_ADDENDUM 1.11;
- EXPANSION_PLAN 1.11;
- указания пользователя: PCIe/MSI отложить; плату программировать только с отдельного разрешения; Vivado запускает пользователь.

**Уровни доказательств:**

| Уровень | Что значит |
|---|---|
| CODE | только код |
| ISA | симулятор ISA |
| RTL | Verilator, RTL == ISA бит в бит |
| FRAME | эталонный fp32-расчёт на CPU |
| ROUTE | Vivado, разводка |
| PHYS | на плате |

Уровни не смешиваются.

**Статусы:** ✅ сделано на указанном уровне, 🔄 делается сейчас, 🟡 частично, ❌ не сделано, ⏸ отложено.

---

## 0. Главное

1. **Новый r8 ещё ни разу не синтезирован.** На плате нет ничего из r8. Последняя разведённая сборка — r7 GRQ `b9fe0ca6`:
   - WNS +0,003 нс, WHS +0,004 нс;
   - LUT 201 711, FF 183 540, BRAM 560,5, DSP 1299;
   - sign-off не закрыт, на плату не загружена.
2. **Работа Claude и Codex разведена по трём местам**: GitHub (`a60ff6f2`), сводная копия Codex (не закоммичена), `openTPU-r7` (не закоммичена). Сборка r8 требует чистого коммита, поэтому **первый шаг — интеграционный коммит**.
3. **На плате работает только старое** (r5, greedy):
   - Qwen3 / 3.5, LFM2, BitNet;
   - Bonsai 4B в старом MLX-формате;
   - MTP k=1, vLLM greedy.

   Всё остальное — максимум ISA/RTL/FRAME.

---

## 1. RTL, ISA и сборка

| № | Пункт | Статус | Доказательства и чей результат |
|---|---|---|---|
| A1 | VOP RAND, сэмплинг, шум, штраф за повтор на карте | ✅ RTL | RTL 4/4 (Claude); генерация на RTL при S=2 1/1 за 300 с (Codex, Docker 12 GiB) |
| A2 | TMM CQ (CQ1–CQ4, LUTSET) | ✅ RTL | RTL 5/5; регрессия TMM 39 passed (Claude) |
| A3 | Sign-off GRQ (CDC, TIMING-54, I/O) | ❌ | Нужен Vivado — пользователь |
| B2 | Сборка r8 m8 133,33 МГц | 🟡 | `-Plan` r7/r8 работает (Codex). С VBFLY хэш `3deb80aa`, но это изменение только в сводной копии Codex. Build не запускался — нужен чистый коммит |
| C1 | QST Q4 (4-битный K) | ✅ RTL | 6/6 |
| C1 | 6-битный V | ⏸ | |
| C2 | PAIR/QUAD + GSC | ⏸ | После замера GRQ на плате |
| C3 | Адамар: VOP BFLY (CAPS2 bit16) | ✅ RTL | RTL == ISA, ISA 3/3 (Claude). Ядро 2048: 6778 → 3827 тактов. Включение в сборку r8 — у Codex, не закоммичено. Ресурсы и тайминг неизвестны |
| C6 | PCIe MSI / litepcie | ⏸ | Решение пользователя |
| C8 | atan в TFN | ✅ RTL | 2/2 |
| C9 | int16-активации | ⏸ | |
| C10 | TMM T5 — плотные триты, 1,6 бита на вес (bit15) | ✅ RTL | RTL 2/2, ISA 8/8. Экономия зависит от формы (Codex): K=2048 — 0 %, K=2560 — 20 %, K=5632 — 18 % |
| C11 | LDX + билинейная выборка | ✅ RTL | RTL 5/5 + 1/1; LDX ~9× быстрее цикла RLD+LD |
| R3 | Fused affine: VOP AFF `(A·P(r))·G(c)+H(c)` (CAPS2 bit17) | ✅ RTL | ISA 7/7; RTL == ISA 2/2 (LN, RMSNorm 3×640). На плате — однопроходная составная функция `se_comp`; округления как MUL+MAC — нормализации бит-в-бит прежние. Регрессия составных функций RTL 59 passed. LLM-путь RMSNorm→матмул уже был слит в QACT. Синтез — нет |
| R4, R6 | MAC, QST T4 | ✅ RTL | |
| R7 | m8 133 МГц | ✅ ROUTE | Только r7 `b9fe0ca6` |
| — | Порядок голов DeltaNet для Bonsai 2 4B/9B: `bonsai2.load_gguf` (перестановка строк при загрузке, Codex `GroupedGDNWeights`) | ✅ RTL | Граф гибридного декодера с перестановкой: ISA (bonsai2+qwen35 96 passed), RTL == ISA 1/1 (токен, логиты и вся DRAM). Реальный 4B через `load_gguf`: « Paris» (FRAME). Ядро `head_order` на устройстве — запасной вариант (RTL 1/1, Codex) |
| — | Опытный декодер Cactus CQ в RTL (`rtl/exp/cactus`) | 🟡 | Синтез вне проекта (Codex); в r8 не входит |
| — | ECC-guard (sticky DED перед scrub) | 🟡 CODE | Тесты есть (Codex); на плате не проверено |
| — | Steering на устройстве | ✅ ISA | 5/5 (Claude) |

---

## 2. FPGA-only: хост не считает

| Пункт | Статус |
|---|---|
| Первый токен ответа на карте | ✅ ISA |
| Uniform, шум, штраф за повтор на карте | ✅ ISA / RTL |
| `--strict-fpga-only` с сэмплингом при VRAND | ✅ (исправлено по ревью Codex) |
| Шум Diffusion / SmolVLA на карте | ✅ ISA (тесты пропускаются без lerobot) |
| Первый токен MTP-префилла | ❌ |
| Позиции до `conv_k − 1` в LFM2 | ❌ |
| MoE полностью без хоста | ❌ |
| Аудит «вычислений на хосте = 0» на уровне backend, негативные тесты | ❌ |
| Исполнитель resident-графа (циклы, fences, IMEM) для YOLOX, Diffusion, SmolVLA | ❌ |
| Планировщик памяти (веса, KV по слотам, lifetimes, отказ до записи при нехватке 4 GiB) | ❌ |
| Многослотовость vLLM, изоляция слотов | ❌ |

---

## 3. Модели

### 3.1. Что есть на диске `C:\ai-models` и в каком формате

| Модель | Формат на диске | Чекпойнт подтверждён? |
|---|---|---|
| Bonsai 2 2B / 4B / 9B | GGUF PTQ1_0, PQ2_0, Q2_0 (+ MTP, DFlash, mmproj) | Да |
| MiniCPM5 2B | GGUF Q4_K_M | Да |
| Granite 4.2 3B / 8B | GGUF Q4_K_M | Да |
| Dory 4B | GGUF Q5_K_M | Да |
| Ouro 1.4B / 2.6B | safetensors (+ int4 v1/v2 в `myOpenTPU\models`) | Да |
| SmolVLA base | safetensors | Да |
| PaddleOCR-VL 1.6 | safetensors | Да; версия не 0.9B |
| Kokoro 82M | `.pth` + голоса | Да |
| MusicGen small | safetensors | Да |
| Cactus needle | контейнер Cactus | Формат CACT не подтверждён |
| BitNet b1.58 2B4T | safetensors + GGUF | Да |
| Qwen3.5-4B Huihui | GGUF Q4_K_M | Да; годится как эталон llama.cpp |
| SDXL, FLUX.2-klein | | Это **не** SD 1.5 |
| FunctionGemma 270M | Только README, веса закрыты | **Нет весов** |
| LFM3 1.7B, ReasonLite 0.6B, SD 1.5 | | **Нет на диске** |

### 3.2. Где каждая модель по уровням

| Модель | CODE | ISA | RTL | FRAME | PHYS | Что осталось |
|---|---|---|---|---|---|---|
| Qwen3 / 3.5, LFM2 | ✅ | ✅ | ✅ | ✅ | ✅ r5 greedy | r8: сэмплинг на карте, первый токен, MTP-префилл, LFM2 позиции |
| BitNet 2B4T | ✅ | ✅ | ✅ | ✅ | ✅ r5 | Полный FPGA-only, сэмплинг на плате, ECC |
| Bonsai (MLX) 4B | ✅ | ✅ | ✅ | ✅ | ✅ r5 (cos 0,99993; ~0,68 ток/с) | Без изменений; это старый путь |
| **Bonsai 2 2B** (GGUF PTQ1_0) | ✅ загрузчик, T5 | ❌ | ❌ | ✅ «Paris» (Claude) | ❌ | Полный граф через T5 на ISA → RTL → плата |
| **Bonsai 2 4B** (GGUF) | ✅ `load_gguf` | ✅ граф (синтетика той же геометрии) | ✅ токен (синтетика) | ✅ «Paris» 13,289 | ❌ | Реальный 4B на ISA (память ~3,55 из 4 GiB, оценка Codex); сверка с Prism runtime |
| **Bonsai 2 9B** | ❌ | ❌ | ❌ | ❌ | ❌ | Адамар-контракт v1 и отдельный head; 4 GiB — под вопросом |
| MTP k=1 | ✅ | ✅ | | | ✅ r5 | k > 1, первый токен MTP |
| vLLM | ✅ плагин | ✅ сэмплинг, logprobs | | | ✅ greedy r5 | Многослотовость, жизненный цикл KV, реальные тесты vLLM |
| YOLOX-Nano | ✅ | ✅ FPGA-only | | | ❌ | Реальный RGB → letterbox → CNN → NMS на плате, IoU/mAP |
| MiniCPM-V | ✅ препроцесс на карте | ✅ | | | ❌ | Ответ по реальной картинке на плате |
| SmolVLA (VLA) | ✅ resident | ✅ tiny | ❌ | ❌ base | ❌ | Настоящий base-чекпойнт, полный цикл Эйлера, качество действий |
| Diffusion Policy PushT | ✅ resident + шум | ✅ | ❌ | ❌ | ❌ | Реальные encoder/U-Net, все шаги, сверка действий |
| Ouro 2.6B | 🟡 int4 v2, MLP (Codex) | 🟡 MLP | 🟡 MLP | ❌ | ❌ | Attention, 48 слоёв × 1–4 прохода, KV по проходам |
| Cactus needle CQ | ✅ TMM CQ | ✅ синтетика | ✅ синтетика | ❌ | ❌ | Сверить реальный контейнер, подключить к графу |
| MiniCPM5 2B | 🟡 деквант K-квантов (Claude), не проверен | ❌ | ❌ | ❌ | ❌ | Сверка FRAME, затем граф (GQA 8:1) |
| Granite 4.2 3B / 8B | 🟡 то же | ❌ | ❌ | ❌ | ❌ | GQA 5:1, head_dim 64 при D=128 |
| Dory 4B | 🟡 то же | ❌ | ❌ | ❌ | ❌ | Архитектура recurrent nGPT, 1/2/4 прохода |
| PaddleOCR-VL 1.6 | 🟡 LDX и билинейная выборка готовы | ❌ | ❌ | ❌ | ❌ | Весь граф |
| Kokoro 82M | 🟡 atan2, шум готовы | ❌ | ❌ | ❌ | ❌ | BiLSTM, duration, iSTFT |
| MusicGen small | ❌ | ❌ | ❌ | ❌ | ❌ | T5, кодбуки, EnCodec |
| SD 1.5 | ❌ | | | | | Нет весов |
| LFM3, ReasonLite | ❌ | | | | | Нет весов |
| FunctionGemma | ❌ | | | | | Нет весов (gated) |
| Steering | ✅ | ✅ | | | ❌ | Для других семейств |
| FFN pruning, speculative decoding k > 1 | ❌ | | | | | |

---

## 4. Проверки

| Набор | Где | Результат |
|---|---|---|
| TMM RTL после T5 | Claude | 39 passed |
| ISA/unit: T5, PTQ1, хост, наблюдаемость, таргеты, CQ, ISA | Claude | 163 passed (1 флейк проходит отдельно) |
| T5, BFLY, PTQ1, steering, CQ, placement | Codex на `a60ff6f2` | 40 passed |
| Cactus, ECC, Ouro, GRQ, PTQ, порядок голов | Codex, сводная копия | 54 passed, 4 skipped |
| Qwen3.5 multilayer, Bonsai, PTQ | Codex | 27 passed |
| Windows-планы сборки | Codex | 26 passed |
| PTQ1 == PQ2 | Claude + Codex | 2B: 151/151; 4B: 201/201 |
| Полный набор, часть 0 | Claude | 412 passed, 33 skipped, **1 failed** (`test_compile_worker_exits_with_its_parent`, причина не найдена) |
| Полный набор, части 1–6 | | **Не запускались** |
| Тесты LeRobot | | Не запускались (нет пакета) |

---

## 5. Что делается сейчас и порядок дальше

### 5.1. Делается

- Интеграция сводной копии Codex в ветку. Ждёт подтверждения пользователя.

### 5.2. Порядок

1. **Интеграционный коммит.** Перенести 43 файла сводной копии в `feature/r7-fpga`, сверяя SHA по manifest V2:
   - review `board.py` / `chat.py` (огромный диф — проверить окончания строк CRLF);
   - не брать `.venv`, модели и сборки;
   - RTL Cactus оставить вне сборки;
   - тесты.
2. **Сборка r8** у пользователя из чистого коммита (VBFLY=1); анализ ресурсов и тайминга (с BFLY и без).
3. **Bonsai 2 4B:** перестановка голов в загрузчик → снять блок → ISA/RTL; эталон Prism для сверки logits.
4. **Bonsai 2 2B** на ISA/RTL через T5.
5. **R3 fused affine:** определить формат, затем ISA, RTL, замер против базового.
6. **FPGA-only инфраструктура:** MTP-префилл, LFM2 позиции, аудит backend, исполнитель графа, планировщик памяти, MoE.
7. **Модели по готовности весов:**
   - MiniCPM5 / Granite / Dory (GGUF);
   - Ouro (attention, проходы);
   - SmolVLA base;
   - YOLOX на плате;
   - PaddleOCR-VL, Kokoro, MusicGen.
8. **Полная регрессия:** разобрать сбой части 0, прогнать части 1–6 и RTL-регрессии с ограничением jobs.
9. **После разрешения прошивки:** физические замеры GRQ=0/1 и все модели на плате.

### 5.3. Кто что делает

**Пользователь:**

- подтвердить интеграцию;
- Vivado: сборка r8, sign-off;
- скачать `bonsai2-4b-mlx`, `lfm3`, `sd15`, `reasonlite`; для FunctionGemma сначала вход в Hugging Face.

**Claude:** пункты 1, 3–8.

**Codex:** review интеграции, эталон Prism для 4B/9B, Ouro/Cactus, RTL-регрессии в Docker.

**Отложено:** C6 PCIe/MSI, C1 6-битный V, C2 PAIR/QUAD+GSC, C9.
