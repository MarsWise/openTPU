# Как сейчас работают BitNet / Qwen / Bonsai: что считает хост и что нужно для скорости (1.11.2026)

Проверено по коду `b9fe0ca6`. Файлы:

- `opentpu/host/chat.py`;
- `opentpu/llm/qwen3.py` (`Engine.generate_card`, `_generate_inputs`);
- `opentpu/llm/generate.py`, `opentpu/host/board.py` (`run_generate`);
- `opentpu/serve/scheduler.py`, `opentpu/vllm/worker.py`;
- `opentpu/llm/placement.py`, `opentpu/bitnet_model.py`;
- `tools/bitnet_*`.

Цифры — из `docs/nextgen/VERIFIED_RESULTS.csv`, `BONSAI_R5_BASELINE_2026-10-31.md`, `docs/benchmarks.md` и `docs/offload.md`.

**Граница по ТЗ.** Хосту можно:

- токены, шаблон, файлы;
- загрузку весов и программ;
- запись входов и чтение выходов;
- перенос байтов без счёта;
- планировщик без численной обработки.

Нельзя: логиты, сэмплинг, случайные числа, любые промежуточные тензоры.

---

## 1. Как идёт один ответ сейчас (`otpu-chat`, Qwen3 / Qwen3.5 / BitNet / Bonsai / LFM2 / Gemma 4)

BitNet и Bonsai идут через тот же `qwen3.Engine`, отдельного движка нет. Шаги ответа:

| Шаг | Где | Что происходит |
|---|---|---|
| 1. Токенизация, chat template | хост | разрешено |
| 2. Prefill промпта | **карта** | `Engine.prefill_chunks`: резидентные prompt runs (ACT_ROWS 8 replay), id токенов вкомпилированы в программы или передаются аргументами; эмбеддинг и RoPE из таблиц образа |
| 3. **Логиты последнего токена промпта → хост** | **хост** | `prefill_chunks` возвращает весь вектор логитов (V × 4 байт: 0,5–1 МБ); `chat._decode_card`: `t = self.pick(logits, self.fed)` — **argmax или сэмплинг первого токена на CPU** |
| 4. Подготовка цикла | хост | запись state block (tok, tpos, left, stop ids, k / T / top-p), разметка `out[]`. При сэмплинге ещё **uniforms из `numpy.random`** и **таблицы штрафа pa/pb по id контекста** (`_generate_inputs`) |
| 5. Декод | **карта** | цикл `LOOP` на устройстве: шаг модели → LM head по чанкам → sink ARGMAX или сэмплер → токен в `out[]` → обратная подача; ведра внимания — через `HALT CHAIN` |
| 6. Чтение токенов | хост | `BoardBackend.run_generate` читает только 64-байтные биты `out[]`, логиты не читает; стоп — словом в state block. Разрешено |
| 7. Детокенизация | хост | разрешено |

**Итог.** Декод на карте полностью FPGA-only при greedy. Первый токен каждого ответа выбирается на CPU — это **во всех LLM в чате, а не только в BitNet и Bonsai** (это уточнение к `FPGA_ONLY_AUDIT` H3).

Поэтому `--strict-fpga-only` сейчас отказывает любому чату (`chat.py`, строка ~1103). При этом `--strict-fpga` (`placement.py`) пишет «token selection: device», если цикл на карте, и первый токен не учитывает.

---

## 2. Все найденные промежуточные вычисления на хосте (по коду)

| № | Что считает хост | Где в коде | Кого касается | Как убрать |
|---|---|---|---|---|
| **H3a** | первый токен ответа: чтение логитов prefill + argmax или сэмплинг | `chat.ask` → `_decode_card` (`self.pick`) | все LLM в `otpu-chat` | **ПО**, см. S1 |
| **H3b** | первый токен в MTP-prefill | `chat._prefill_mtp` (`pick=lambda lg: self.pick(...)`), `mtp.prefill` | Qwen3.5 + MTP | ПО: S1 для MTP-префилла |
| **H3c** | первый токен в `otpu-serve` | `serve/scheduler.py` (`pick(lg)`, сэмплинг с uniform хоста) | сервер | ПО: S1 |
| **H3d** | первый токен и все токены с «неустройственным» сэмплингом в vLLM без `fpga_only` | `vllm/worker.py` (`prefill_chunks(..., logits=last)` → хостовый `sample`) | vLLM | уже есть режим `fpga_only` (последний токен промпта идёт через устройственный шаг); сделать режимом по умолчанию для strict |
| **H1** | случайные числа для сэмплинга | `qwen3._generate_inputs` (`rng.random`), `vllm/worker.py` (`uni = r.rng.random()`), scheduler | всё с temperature > 0 | **RTL `V_RAND`**: экспериментальный Philox4x32-10 уже есть в `rtl/exp/philox`, не подключён к ISA |
| **H10** | таблицы repetition penalty (1/R и R по id контекста) | `qwen3._generate_inputs` (pa/pb по `np.unique(context)`) | сэмплинг со штрафом | **ПО**: устройство уже обновляет таблицы для своих токенов; для промпта — цикл `RLD` по id контекста (запись R и 1/R по адресу) |
| **H11** | сэмплинг, который устройство не берёт (top-k > 64, min-p, presence/frequency penalty, logit_bias и т. п.) — хост читает логиты **каждого** токена | `chat.on_card = False` → `eng.step` + `self.pick`; vLLM `eng.run_rows` + `sample` | чат и vLLM с такими параметрами | ПО: расширить сэмплер (min-p — CMP/SEL от max; penalties — таблица счётчиков; top-k > 64 — порог в несколько проходов) или отказ в strict |
| **H12** | позиции < `conv_k − 1` в `generate_card` | `qwen3.generate_card`: `if p < self._conv_lo: lg = self.step(tok); np.argmax / reference_pick` | LFM2, Qwen3.5 (только самое начало контекста) | ПО: в strict отказ (промпт с шаблоном всегда длиннее) или программы с run-time позицией от 0 |
| **H13** | эмбеддинг и RoPE-строки в per-position режиме | `Engine` при `device_inputs = False` (`--per-position`, не resident) | любой при `--per-position` | уже помечено HOST в `placement.py`; в strict запрещено |
| **H14** | **гибридный BitNet:** RMSNorm, RoPE, softmax attention, relu², LM head на numpy; на карте только TMM | `opentpu/bitnet_model.py`, `tools/bitnet_generate.py` (строка `9ec42b59` в CSV «ternary TMM + host ops») | старые замеры BitNet | **не FPGA-only.** Пометить legacy, не использовать как доказательство. FPGA-only путь — `tools/bitnet_engine.py --mode generate` / `otpu-chat` (кроме H3a) |
| H15 | логиты для отладки | `gen_debug`, `bonsai2_validate.py`, `--compare` | проверки | разрешено как проверка, не инференс |

**Только перенос байтов (разрешено ТЗ):**

- строки PLE Gemma 4 E4B (`_host_rows`);
- int8-строки эмбеддинга (`embed_host`);
- эксперты MoE из пула (`host/offload.py`; router и top-k на карте);
- программы и state block.

**Отдельно не FPGA-only:** resident-графы VLA и Diffusion (H2 — нормализация входов и шум на хосте, H6 — запуск программ по одной). Подробно — `FPGA_ONLY_AUDIT_2026-11-01.md`.

### Как убрать (ПО)

**S1 — первый токен на карте.**

- *Быстрый вариант* (как уже сделано в vLLM `fpga_only`): prefill `ids[:-1]` без LM head, затем `generate_card(ids[-1], n)`. Цикл сам подаёт последний токен промпта, и его LM head выбирает первый токен ответа на карте.
  - В `chat._decode_card` путь `isinstance(logits, int)` уже есть (resume).
  - Цена: один токен промпта идёт шагом декода, а не в чанке prefill. Это +1 проход весов к TTFT: ~50–80 мс у BitNet и Qwen3.5-2B.
  - Аккуратно со снапшотом `at`, MTP и строками изображения.
- *Полный вариант*: последний чанк prefill компилируется с `m.lm_sink` сэмплера (как в `generate.py` и batch steps). Он кладёт токен в `S_TOK` / `out[p]` и через `HALT CHAIN` переходит в программу декода — один запуск на весь ответ.

**S2.** Инициализация штрафа на карте (H10).

**S3.** Расширение сэмплера (H11) или явный отказ в strict.

**S4.** `placement.py`: добавить стадии «первый токен», «uniforms», «penalty init», `_conv_lo`, чтобы `--strict-fpga` их видел. Сейчас стадия «token selection: device» вводит в заблуждение.

**S5.** `--strict-fpga-only` разрешить, когда выполнены S1, S2 и (для сэмплинга) `V_RAND`. Для greedy достаточно S1 и S4.

---

## 3. Скорость: где мы сейчас (физические замеры)

| Модель | Формат | ток/с (device) | Ограничение |
|---|---|---|---|
| Qwen3-0.6B | int8 / fp4 + int8 head | 24.76 / 35.81 | DRAM |
| Qwen3.5-0.8B | int8 / fp4 | 19.92 / 27.61 | DRAM + рекуррентность DeltaNet на VPU |
| Qwen3.5-2B | int8 / fp4 + fp4 head / + MTP | 8.39 / **15.20** / **19.1** | DRAM, ~95 % |
| LFM2.5-230M | int8 / fp4 | 66.8 / 96.25 | DRAM |
| BitNet 2B4T, b=1 | ternary + head int8 / fp4 | 17.52 / **20.90** (wall), device 55.8 / 46.5 мс | DRAM; head int8 — ~⅓ байтов токена |
| BitNet vLLM | 4 / 8 слотов | 27.58 / 28.92 всего | TMEM (4 слота), **ECC-ошибки DDR канала 0 под нагрузкой** |
| **Bonsai 2 4B** (r5) | ternary GSC | **0.68** | **17.9 ГБ чтений на токен** (веса ~1 ГБ): GSC перечитывает чанк 4×; MXU_MAC 17 % |

Условия:

- r5-m8, 137.5 МГц; Bonsai — `45b1c1d3`, 133 МГц.
- Карта читает DRAM на ~14 ГБ/с при декоде (теоретический пик 2 × DDR3-1066 ≈ 17 ГБ/с).
- **Декод при b=1 упирается в память:** ток/с ≈ 14 ГБ/с / байты на токен. Qwen и BitNet уже на 90–98 % этого потолка. Хост почти ничего не съедает: у BitNet wall 17.52 против device ~17.9.
- **Bonsai — единственная модель далеко от потолка:** ~0.7 против ~10–12 ток/с (оценка).

---

## 4. Что нужно для скорости

### 4.1 Без нового RTL (ПО и конфигурация)

| № | Что | Эффект | Статус и условие |
|---|---|---|---|
| P1 | **LM head 4-бит** (fp4 / int4) по умолчанию, где проходит точность | BitNet 17.5 → 20.9 (+19 %), Qwen3.5-2B 12.6 → 15.2 (замер) | у BitNet с fp4 head только первые 8 из 128 токенов совпали с int8 — нужна проверка логитов против HF |
| P2 | **MTP / speculative** | Qwen3.5-2B 15.2 → 19.1 (замер) | для BitNet, Bonsai и Qwen3 MTP-слоя нет: драфт-модель или self-speculative (пропуск слоёв) с проверкой на карте; сэмплинг — после `V_RAND` |
| P3 | **Батч (vLLM / serve)** | BitNet 1 → 4 слота: 14.5 → 27.6 ток/с суммарно | ограничено TMEM (4 слота) и ECC канала 0 — сначала надёжность DDR |
| P4 | **prompt runs для Bonsai** (ACT_ROWS 8 replay, GRQ берёт M ≤ MCOLS) | prefill ~в 8× меньше проходов весов | baseline Bonsai снят с `--no-prompt-runs`: 27 с на 17 токенов |
| P5 | **кэш программ и прогрев** | у Bonsai 1.65 с компиляции до первого токена | `prog_cache`, `warm()` |
| P6 | S1 (первый токен на карте) | убирает чтение 0.5–1 МБ логитов и цикл хоста | нужно для FPGA-only, на скорость влияет мало |
| P7 | vLLM: несколько шагов на устройстве за один вызов планировщика (`HALT CHAIN`) | меньше запусков хостом на токен | конфликтует с пошаговой моделью vLLM; только для greedy |
| P8 | **KV 8 бит** (есть) на длинных контекстах | KV-доля трафика ×0.5 | для 4K+ контекстов |

### 4.2 RTL (по убыванию пользы)

| № | RTL | Для чего | Ожидание (оценка) | Статус |
|---|---|---|---|---|
| **R-GRQ** | TMM GSC GRQ: каждый чанк читается один раз | Bonsai 2 (4B, 9B) | GEMV 2560×2560 349k → ~51k тактов (~6.8×) | **RTL SIMULATED + собран** (`7ecc66e3…`, `cfg_hash d9b863d5`); нужны sign-off и загрузка с вашего разрешения, затем замер `decode_profile.py` |
| C2 | GRQ «QUAD»: 4 четверти за такт с раздельными масштабами | Bonsai, если после GRQ MXU_MAC остаётся низким | до ~2–4× поверх GRQ на MXU-части | после замера GRQ |
| C3 | блок Адамара | Bonsai: преобразование активаций на VPU | по профилю после GRQ | условно |
| C10 | декодер плотных тритов (1.6–1.8 бит вместо 2-битных слотов) | BitNet, Bonsai 9B PTQ1_0 | −10…20 % байтов → +10…20 % ток/с (DRAM-bound) | условно |
| C1 | KV 4/6 бит (`QST` 4/6 + int6 в MXU) | длинный контекст: Qwen, VLM | у upstream V 6-бит проходит, K 4-бит ломает все модели | условно |
| R3 | fused affine VOP (RMSNorm / LN / GN: 6 проходов → 3) | VPU-доля: ViT, SD, VLA; в LLM-декоде мала | по профилю | PLANNED |
| R6 | `QST` с транспонированной записью V^T | attention prefill (VLM, SD, длинные промпты) | по профилю | PLANNED |
| — | больший TMEM (BRAM) | больше слотов батча (сейчас 4) | ×(слоты) до compute-bound | конфигурация + бюджет BRAM |
| — | **надёжность DDR канала 0** (калибровка, ECC под нагрузкой) | без неё нельзя поднимать батч и полосу | — | блокер: DED при 4–8 слотах и при fp4-перфе |
| `V_RAND` | генератор на VPU | FPGA-only сэмплинг (не скорость) | — | обязательный для FPGA-only |

**Чего нет и не будет на этой плате:** полосы памяти больше ~14–15 ГБ/с. Оба канала DDR3-1066 уже используются, а контроллер даёт ~82 % пика. Поэтому для LLM-декода b=1 выигрыш даёт только уменьшение байтов на токен (форматы, head, KV, триты) и несколько токенов за проход (MTP, speculative, батч). Тактовая частота и MXU важны только для prefill, батча и Bonsai.

---

## 5. Порядок работ (предложение)

1. **S1 быстрый вариант + S4** (ПО, 1–2 дня): первый токен на карте для greedy во всех LLM; `--strict-fpga-only` для greedy. Проверка в ISA и на карте вашими скриптами.
2. **Bonsai GRQ:** sign-off → ваше разрешение на загрузку → `decode_profile.py` до и после, проверка логитов.
3. **P1** head 4-бит с проверкой логитов против HF (BitNet, Qwen3.5).
4. **`V_RAND`** (Philox из `rtl/exp/philox` → VOP, seed и счётчик в state block) + S2 → FPGA-only сэмплинг, затем speculative с сэмплингом.
5. Надёжность DDR канала 0 → батч 4–8 слотов, затем TMEM.
6. По замерам: C2, C3, C10, C1, R3, R6.
