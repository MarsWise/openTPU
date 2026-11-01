# Передача Codex: состояние r8 на 1.11.2026, ночь (Claude)

## 1. Где всё

**Ветка:** `MarsWise/openTPU:feature/r7-fpga`, HEAD `6023000d`. Всё ниже закоммичено и запушено. PR #1 открыт, draft, не слит.

Сводный статус по всем ТЗ: `docs/nextgen/TZ_FULL_STATUS_2026-11-01_RU.md`, обновлён в `442f92e8`.

**Плата:** не программировалась, Vivado не запускался.

## 2. Что сделано после `a60ff6f2`

| Коммит | Что | Доказательства |
|---|---|---|
| `63df12ad` | Перенесены из сводной копии Codex (`claude-r8-review-a60ff6f2`): `kernels/head_order.py`, `llm/gdn_head_order.py` с тестами, fail-closed `tools/bonsai_ptq1_check.py`, `tools/bonsai2_gguf_check.py` (`--grouped-out`, `--grouped-inputs`), VBFLY в `build_r5_timing.ps1` / `build_r8_133.ps1` / `r8_build.cmd`, `tools/profile_r3_baseline.py`, `test_r8_build_plan_windows.py`. Перевод строк приведён к LF, как в репозитории | Тесты head-order / PTQ / BFLY: 36 passed |
| `8268044e` | **R3: VOP AFF** (func 30, CAPS2 bit17 `VAFF`) — см. раздел 3 | ISA 7/7; RTL == ISA 2/2 |
| `8268044e` | **Bonsai 2 4B/9B:** `bonsai2.load_gguf(path)` для GGUF с `gdn_v_grouped` переставляет строки V-голов (QKV-V, z, a, b, conv, A, dt) в grouped-порядок при загрузке через `GroupedGDNWeights` Codex. `ssm_out` не трогается. Устройство идёт по проверенному grouped-пути (как MLX 4B на r5). Блок снят только при наличии `gdn_v_grouped`. 9B (контракт v1) по-прежнему отклоняется | Реальный 4B PTQ1_0 через `load_gguf`: « Paris» 13,289 (FRAME, CPU). bonsai2 + qwen35 ISA: 96 passed. Новый RTL-тест `test_grouped_gguf_wrapper_native_hybrid_rtl`: токен гибрида (2 V на K, attention, tied head) — логиты и вся DRAM RTL == ISA |
| `442f92e8` | Статус обновлён | Регрессия составных функций на RTL (`test_se_comp`, `test_tfn`, `test_fp_rtl`) после правки `se_comp`: 59 passed (2 первичных сбоя были «No space left on device», после чистки прошли) |
| `b30f5ea0`, `ab37e272`, `6023000d` | **WIP:** `llm/gguf_llama.py` + `tools/gguf_llama_check.py` — загрузчик GGUF архитектур llama/granite (MiniCPM5, Granite 4.2) из K-квантов под имена HF для `qwen3`. Снятие перестановки q/k — опционально (`--qk-permuted`). Множитель attention Granite вшит в q | MiniCPM5 Q4_K_M: « Paris» без снятия перестановки (FRAME). **Granite: результат не записан, не проверен.** ISA/RTL — нет |

## 3. R3 VOP AFF — подробно

**Семантика:** `dst(r,c) = (A(r,c)·P(r))·G(c) + H(c)`, где:

- P — слово строки (bmode ROW);
- `G = T[imm + c]`, `H = T[imm + cols + c]`; флаг DONE означает H = −0;
- `cols ≤ 256`;
- G и H читаются в буферы столбцов до записи.

**RTL:**

- `otpu_pkg` (V_AFF, CC_AFF, reads_b, диапазоны hazard);
- `otpu_vpu` (fill: G в C-буфер, H в D-буфер; `n_slots = 2`; $fatal без stream engine);
- `otpu_se_comp`: однопроходная составная функция, S0 `v = x·P`, S1 `v = v·G + H`; новые входы `in_g`/`in_h`.

**Компилятор и ядра:**

- `Builder.aff`, `Builder.aff_params` (упаковка G|H раз на ядро), `ol.aff`, `ol.has_aff`;
- `eltwise.layer_norm` / `rms_norm`, `vit.layernorm_`, `lib.rmsnorm`.

Округления — как у MUL + MAC, поэтому нормализации бит-в-бит прежние.

**Выигрыш:**

- RMSNorm и LayerNorm с несколькими строками: один проход вместо двух на аффинную часть.
- Для одной строки LN оставлен старый путь: копии упаковки не окупаются.
- В LLM-пути RMSNorm → матмул уже был слит в QACT (RSCALE/CSCALE), там AFF ничего не меняет.

**Не сделано:** синтез, ресурсы и тайминг. AFF использует существующие стадии `se_comp`, новых умножителей нет.

## 4. Что из сводной копии Codex НЕ перенесено (остаётся у Codex)

- Ouro: `llm/ouro.py`, `kernels/ouro.py`, `tools/ouro_*`, тесты.
- Cactus: `opentpu/cactus/`, `kernels/cactus_cq.py`, `rtl/exp/cactus/`, тесты.
- ECC-guard: `host/ecc_guard.py`, патчи `board.py` / `chat.py`, `test_ecc_guard.py`.
- Правки `build.tcl`, `decode_profile.py`, `bonsai2_validate.py`, `vllm_smoke.py`, скриптов r7 card/sim tests.

Их нужно перенести поверх `6023000d`:

- **конфликтуют:** `tests/test_bonsai2.py` (у меня добавлены RTL-тест и параметр `rtl`), `tests/test_qwen35.py`, `tools/bonsai2_gguf_check.py` (добавлен `--loader`), `tools/build_r5_timing.ps1` (добавлен VAFF);
- **`board.py`:** у меня добавлены проверки CAPS2 для VBFLY и VAFF; у Codex — ECC.

`board.py` и `chat.py` сводить по смыслу. Огромный диф у Codex — проверьте CRLF.

## 5. Важно для сборки r8

- В r8 теперь включены VBFLY (bit16) и VAFF (bit17):
  - `OTPU_VBFLY_GEN` / `OTPU_VAFF_GEN = 1` в `build_r5_timing.ps1` при `-R8`;
  - CFG_HASH считается с `VBFLY=True, VAFF=True` (план Codex `3deb80aa` был без VAFF — **хэш изменился**, пересчитать `-Plan`);
  - `r8_build.cmd` проверяет VAFF в `otpu_ctrl.sv`.
- В VPU логика BFLY и AFF не отключается параметром. Влияние на LUT/FF/timing покажет только синтез, в том числе при r7.
- Скрипт сборки требует чистый коммит: после переноса работы Codex закоммитить, затем `-Plan` и Build.

## 6. Проверки — что прогнано и что нет

**Прогнано (контейнер, Verilator 5.046):**

| Набор | Результат |
|---|---|
| AFF ISA | 7/7 |
| AFF RTL | 2/2 |
| BFLY ISA / RTL | 3/3, 1/1 |
| T5 RTL | 2/2 |
| TMM RTL | 39 |
| Составные функции RTL | 59 |
| Граф Bonsai с перестановкой голов RTL | 1/1 |
| bonsai2 + qwen35 ISA | 96 |
| target / host / observability / aff / bfly ISA | 127 + 1 тайминговый флейк (отдельно проходит) |

**Не прогнано:**

- полный набор: части 1–6 (часть 0: 412 passed, 1 failed `test_compile_worker_exits_with_its_parent`, не разобран);
- RTL-регрессия всего дерева после AFF / BFLY;
- тесты LeRobot (нет пакета).

**Ограничение контейнера:** диск под `build/verilator` переполняется. Чистить старые сборки перед RTL-прогонами, один Verilator за раз.

## 7. Открытые задачи (по порядку)

1. Перенести Ouro / Cactus / ECC поверх `6023000d`, review `board.py` / `chat.py`, чистый коммит → `-Plan` r8 → Vivado (пользователь).
2. `gguf_llama`: проверить Granite на FRAME; MiniCPM5 / Granite — tiny ISA и RTL той же геометрии (GQA 8:1 и 5:1, head_dim 64 при D=128).
3. Bonsai 2: реальный 2B/4B на ISA (память 4B ~3,55 GiB); 9B — контракт v1 и отдельный head; сверка логитов с Prism runtime.
4. FPGA-only:
   - позиции LFM2 до `conv_k − 1`: сейчас они идут отдельными программами с позицией, известной при компиляции (`compile_prompt_run p0`); нужно проверить, остаётся ли там счёт на хосте;
   - первый токен MTP-префилла;
   - MoE без хоста;
   - аудит backend и негативные тесты;
   - исполнитель графа;
   - планировщик памяти;
   - многослотовость vLLM.
5. Модели: Ouro (attention, проходы), SmolVLA base, YOLOX на плате, PaddleOCR-VL, Kokoro, MusicGen; после скачивания — LFM3, SD 1.5, ReasonLite, FunctionGemma (gated).
6. Полный набор тестов, части 1–6, и RTL-регрессия.

**Отложено пользователем:** C6 PCIe/MSI. **Не трогать:** FeSens/openTPU, main форка, PR #1. **Плата:** только с разрешения.
