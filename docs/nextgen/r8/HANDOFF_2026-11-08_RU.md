# Передача: сессия Claude 8.11.2026 — всё для продолжения в новой сессии

Читать вместе с:

- `docs/nextgen/r8/STATUS_2026-11-08_RU.md` — статус, замеры, разделы 2.1–2.11;
- `docs/nextgen/r8/MODELS_TODO_2026-11-08_RU.md` — план по моделям.

Копии лежат в проекте claude.ai «myOpenTPU»:

- `claude/HANDOFF_2026-11-08.md`;
- `claude/STATUS_2026-11-08.md`;
- `claude/MODELS_TODO_2026-11-08.md`.

## 1. Git

| Что | Значение |
|---|---|
| Ветка публикации | `MarsWise/openTPU:feature/r7-fpga` |
| HEAD на конец сессии | `876b6282` (с этим документом — следующий коммит) |
| PR #1 | open / draft / не слит (проверять `gh api repos/MarsWise/openTPU/pulls/1 --jq '[.state,.draft,.merged]'`) |
| Запреты | push только в `feature/r7-fpga`; никогда FeSens/openTPU, main форка, PR #1 не трогать; без force-push; не удалять незакоммиченное Codex в `C:\myOpenTPU\openTPU-r7` (27 файлов, HEAD `b9fe0ca6`) |
| Подпись коммитов | `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` + строка `Claude-Session: …` текущей сессии |

**Как пушили в этой сессии (контейнер):**

- рабочее дерево: `/tmp/claude-0/-home-claude/d587f972-…/scratchpad/r7wt`, ветка `r7`;
- push из `/home/claude/opentpu` командой `git fetch <r7wt> r7 && git push -q origin FETCH_HEAD:refs/heads/feature/r7-fpga`.

**В новой сессии** контейнер будет новый:

- склонировать `MarsWise/openTPU`, ветку `feature/r7-fpga`, работать в ней;
- remote проверить перед push. GitHub отвечает «repository moved» на тот же URL — это безвредно.

**Коммиты этой сессии** (поверх `c0026f57`):

| Коммит | Что |
|---|---|
| `06cbd9ab` … `4b5e9b01` | Перенос работы Codex, 6 коммитов: scoreboard 2^32; strict-исправления; планировщик (резерв, DeviceArena); Ouro / Cactus / ECC; скрипты сборки; EXPERIMENTAL INT6 в `rtl/exp` |
| `800ae9d0` | `otpu_seq`: стадия T между Q и C (путь fp_seg → hz_coarse) |
| `8f56da69` | **C1: 6-битный K** — QST Q6, CAPS2 bit18 `QST6`, три 2-битные плоскости, раскладка TMM CQ2 GRQ; `kernels/kv6.py` |
| `9550518a` | **C1: 6-битный V** — QST Q6T (флаг bit6), четыре токена на байт через путь T4 |
| `ab02e915` | **VOP CSWP** (func 31, CAPS2 bit19 `VCSWP`): битонный шаг; `kernels/sort.py` (`bitonic`, `cumsum`) |
| `c840cb3f` | Gemma 3 (FunctionGemma) на программе gemma4: `Spec.from_gemma3`, `Gemma3Weights`, `v_norm` |
| `d5c71bae` | статус |
| `671e97d6` | `kernels/lstm.py` + замер на RTL |
| `3d416ed0` | `impl_directives.tcl`: опции площади синтеза; C9 закрыт |
| `1808ebc6` | статус: 5 пунктов FPGA-only |
| `876b6282` | `MODELS_TODO` |

## 2. Что сделано и доказано

Уровни доказательств не смешиваются: ISA / RTL (Verilator 5.046, бит в бит) / FRAME / SYNTH.

**RTL:**

| Что | Доказательство |
|---|---|
| Codex: свои RTL-тесты | 26 passed |
| QST6 K | ISA 6/6, RTL 4/4 |
| QST6 V | ISA 4/4, RTL 3/3 |
| CSWP | ISA 4/4 + cumsum 3/3, RTL 3/3 |
| Регрессии | 75 passed (quant / TMM / seq), 39 passed (quant / T4), 39 passed (VPU), 36 passed (gemma4 ISA) |
| Gemma 3 | токен RTL == ISA на маленькой модели (позиция 520, окно) |

**Модели (реальные веса):**

| Модель | Результат |
|---|---|
| FunctionGemma 270M | FRAME == HF (2e-4, « Paris»); ISA int8: « Paris.\n\nParis is the» == golden W+A; порог KL в `validate.py` не пройден — golden не моделирует точки квантования gemma4 для Gemma 3 |
| ReasonLite 0.6B (Qwen3) | ISA: 1-й токен == HF, cos 0,997; 2-й расходится (« French» / « fe», cos 0,996); полный golden не влез в 7 ГБ |

**Замеры, закрывшие RTL-пункты:**

- **Слитый шаг LSTM — не нужен.** H = 256: шаг 2386 тактов, из них VPU 115 (5 %). Шаг упирается в поток W_hh.
- **C9 int16 — не нужен.** FRAME SD 1.5, 256 px, 20 шагов DDIM, 1 сид: W8A8 — 17,7 дБ, W8 + hi+lo — 18,2 дБ против fp32. Сцена та же. Скрипт: `tools/nextgen/research/sd15_precision.py`. Суть: перехват Linear / Conv2d / SDPA в torch, int8 на 128-блок, обязательно учитывать `attn_mask` / `is_causal`.

**Синтез r8** (пользователь, Vivado 2024.2.2, `d5c71bae`, режим Synth):

- core_clk 133,33 МГц: WNS **+0,228 нс** с запасом 0,15 нс;
- итоговые −0,142 нс — 6 путей внутри IP XDMA (SRL сброса CPLL);
- hold после синтеза не показателен.

| Ресурс | Занято | Доля |
|---|---|---|
| LUT | 270 336 | **90,5 %** |
| FF | | 33 % |
| BRAM | | 61 % |
| DSP | 1302 | 68 % |

Потребители LUT: MXU 82,5k (из них 11,5k SRL), VPU 58,6k, LiteDRAM 23k, XDMA 19k, TMEM 15,8k, `otpu_tmm_gsc` 11k при 527 FF (подозрительно), quant 10,2k, seq 9,1k.

**CFG_HASH r8 = `93e29b18`** (с VCSWP). Хэши `e0d5adb4` и `3deb80aa` недействительны.

## 3. Ждём от пользователя

1. `C:\myOpenTPU\downloads\get_r8.cmd` — чистая копия `C:\myOpenTPU\openTPU-r8` на HEAD ветки.
2. **До** нового синтеза: `vivado.bat -mode batch -source C:\myOpenTPU\downloads\r8_util_deep.tcl`. Пишет `reports\deep_util_hier.rpt`, `deep_prims_*.txt`, `deep_fanout.rpt`.
3. `r8_build.cmd Synth` с новыми опциями (LUT combining, resource sharing). Сравнить LUT и WNS.
4. Если LUT заметно ниже и WNS > 0 — полная сборка `r8_build.cmd`.

## 4. Следующая работа (по порядку)

1. **Площадь r8** по `deep_util_hier.rpt`, без отключения BFLY / AFF / CSWP:
   - длинные SRL → BRAM;
   - короткие SRL → FF;
   - async FIFO `otpu_mem_ch` → BRAM;
   - источник 11k LUT в `otpu_tmm_gsc`;
   - мультиплексоры TMEM pump.

   После каждой правки — RTL == ISA и регрессия.
2. **6-битный KV в графе внимания qwen3** (раздел 2.11 статуса).
3. **MiniCPM5 → Granite 3B** на ISA с реальными весами (лениво по тензору).
4. **LFM3** (`llm/lfm3.py`), затем **Bonsai 2 2B/4B** на ISA.
5. **Первый токен MTP**, затем **позиции LFM2**.
6. **Исполнитель графа (YOLOX)** и **планировщик памяти.**
7. **Cactus → Ouro → Dory.**
8. **Kokoro → SD 1.5 → MusicGen → PaddleOCR-VL.**

Ещё по ТЗ: сверка с upstream FeSens; полный набор тестов (части 1–6); полная RTL-регрессия; обновление `docs/nextgen` (PROGRESS, MODEL_COMPATIBILITY, OPERATOR_COVERAGE_MATRIX, VERIFIED_RESULTS.csv).

**Отложено:** C2 (после замера GRQ на плате), C6 PCIe MSI (решение пользователя), A3 sign-off GRQ (Vivado, пользователь).

## 5. Практика: окружение и процедуры

**Контейнер:**

- Verilator 5.046 в `/opt/verilator/bin`; запускать `PATH=/opt/verilator/bin:$PATH`;
- iverilog есть; transformers 5.20, torch, diffusers 0.42 — есть;
- 7 ГБ RAM — один тяжёлый процесс за раз;
- диск тесный: перед RTL-прогонами `find build/verilator -maxdepth 1 -mindepth 1 -mmin +12 -exec rm -rf {} +`;
- slow-тесты — с `--runslow`.

**Новая возможность CAPS2:**

1. `tools/nextgen/research/addcap.py REPO NAME BIT PREV "desc"`.
2. Затем вручную:
   - поле `Config` и env в isasim;
   - `d.pop` в `test_target`;
   - словари caps в `test_host` и `test_observability`;
   - проверка при загрузке в `board.py`;
   - `OTPU_*_GEN` и `CFG_HASH` в `build_r5_timing.ps1`;
   - проверка в `r8_build.cmd`;
   - `test_r8_build_plan_windows.py`.

**Веса с диска C в контейнер:**

1. В VM (`device_bash`): `dd`-нарезка по 380 МБ в `C:\myOpenTPU\reports\_<model>_parts`.
2. `device_stage_files` (≤ 400 МБ на файл).
3. Склейка `cat` в `/home/claude/models/<model>`.

Временные папки `_reasonlite_parts`, `_fgemma_parts`, `_sd15_parts` (~4 ГБ) можно удалить: удаление в VM требует разрешения пользователя, оригиналы в `C:\ai-models`.

**Модели на диске C:**

| Путь | Модель |
|---|---|
| `C:\ai-models\hf-specialized-2026\{lfm3-omni,reasonlite}` | LFM3, ReasonLite |
| `agents\FunctionGemma_270M` | FunctionGemma |
| `small\{MiniCPM5_2B,Granite_4.2_3B,Granite_4.2_8B}` (GGUF) | MiniCPM5, Granite |
| `dory-4b` | Dory |
| `ocr\PaddleOCR_VL_1.6` | PaddleOCR-VL |
| `audio\{hexgrad--Kokoro-82M,facebook--musicgen-small}` | Kokoro, MusicGen |
| `diffusion\sd-v1-5` | SD 1.5 (скачана 8.11, SHA-256 сверены, ревизия `451f4fe1`) |
| `image\{SDXL,FLUX.2-klein}` | не помещаются в 4 ГБ DDR3 платы |
| `ouro-*`, `cactus` | Ouro, Cactus |
| `ternary` | Bonsai 2 |

**Скрипты пользователя** в `C:\myOpenTPU\downloads`:

- `download_models.py`, `download_sd15.cmd`;
- `get_r8.cmd`;
- `r8_util_deep.tcl`.

**Безопасность:**

- пароль хоста из описания проекта никуда не записывать;
- плату не программировать, Vivado-реализацию не запускать без разрешения — сборки делает пользователь;
- код из папок моделей (`modeling_*.py`) не исполнять — читать и переписывать.
