# r8 на плате: хост-часть и порядок запуска (8 ноября 2026)

Сборка: `vivado-r8-m8-133.33-a4138552` (исходник a4138552, m8, 133.33 МГц, CFG_HASH `93e29b18`).
Битстрим `otpu.bit` появляется только если setup и hold прошли. Пока сборка не закончена, ничего
не загружаем.

## Что подготовлено

| Файл | Где запускается | Что делает |
|---|---|---|
| `tools/nextgen/windows/r8_load.cmd` + `r8_load.tcl` | Windows (JTAG) | Показывает SHA256 битстрима, спрашивает `yes`, грузит его в FPGA **только в SRAM (volatile)**, без flash. Проверяет одну цель JTAG, part xc7k480t и DONE=1. |
| `tools/nextgen/windows/r8_restore_pcie.cmd` | Windows → ssh на 192.168.0.157 | `setup_pcie.sh --rescan` (нужен sudo-пароль в окне), затем печатает build / core_khz / CFG_HASH и `r8` или `NOT r8`. |
| `tools/nextgen/windows/r8_card_tests.cmd` + `card_runs/r8_card_tests.sh` | Windows → хост | Весь прогон r8 на карте, лог и tgz сохраняются рядом с cmd. Ничего не грузит, PCIe не трогает. |
| `tools/r8_card_check.py` | хост | Идентичность (CFG_HASH, 16 бит CAPS2 r8), все одно-инструкционные проверки otpu-diag, многоинструкционные случаи r8 (TMM CQ1–4 / GSC / GRQ / T5, QST Q4 / Q6 / Q6T, CSWP-сортировка, RAND, LDX). Каждый случай сравнивается с ISA бит в бит: DRAM и весь TMEM. |
| `tools/r8_kv_check.py` | хост | Маленький Qwen3 с KV int8 / kv6 / k6v6: промпт 261 токен, шаги с хоста, `generate_card` на карте, логиты, токены и весь образ DRAM, карта == ISA. |
| `firmware_contract.json` | — | Запись r8 (cfg_hash 0x93e29b18; build_id и sha256 `pending` до готового битстрима). |

Порядок `r8_card_tests.sh`:
1. sync из `sync.bundle`.
2. Проверка, что в дереве есть a4138552.
3. Остановка, если загружена не r8.
4. Шаги A (`r8_card_check`) и B (`r8_kv_check`).
5. Модельные шаги r7: Diffusion, SmolVLA, YOLOX strict FPGA-only, vLLM sampled / greedy / logprobs, Bonsai GRQ off/on.

ECC проверяется до и после каждого шага.

## Порядок (каждый шаг запускает пользователь)

1. Дождаться конца сборки и проверить `reports\SUMMARY.txt`: WNS ≥ 0, WHS ≥ 0, route полный.
2. Остановить все прогоны на карте. Запустить `r8_load.cmd` и на вопрос ответить `yes`.
3. Запустить `r8_restore_pcie.cmd`. Должно быть напечатано `cfg hash 0x93e29b18 r8`.
4. Запустить `r8_card_tests.cmd` (4–6 часов) и прислать `r8_card_tests.log` и `.tgz`.

Откат: перезагрузка питания возвращает образ из flash (r7). Затем `r8_restore_pcie.cmd`: он
напечатает `NOT r8`, это ожидаемо.

## Найдено при подготовке: ошибка хоста в wait-записях VOP AFF (исправлено)

Прогон `r8_kv_check --sim` на модели платы (Verilator 5.046) остановился на ассерте
sequencer'а: `otpu_seq: WAITC: pc 10's wait 0 (k 4) misses slot 3 (pc 5)`. Модель была собрана
с конфигурацией r8 (хэш 93e29b18, WIN 16, SEQ_WC 1).

**Причина.** В `opentpu/waits.py footprint()` не хватало двух чтений VOP AFF:
- таблицы G/H по `R[rd] + gh` (2·cols слов, с DONE — cols);
- операнда B (масштаб строки, ROW).

В RTL (`otpu_pkg.sv`) оба чтения есть. Из-за этого хост выдавал AFF в RMSNorm счётчик DMA,
который не ждал LD таблицы.

**Почему это важно.** С WAITC sequencer на плате не сравнивает диапазоны. AFF прочитал бы
таблицу до конца LD, и на r8 результат был бы неверным без всякой ошибки. На r7 AFF нет, там
этого не было.

**Исправление.** `footprint()` теперь совпадает с otpu_pkg. Добавлен тест
`tests/test_waits.py::test_aff_waits_for_its_scale_and_gh_tables`: без исправления он падает,
с исправлением проходит. После исправления та же модель платы проходит шаги.

Исправление только в хосте: битстрим пересобирать не нужно.

## Проверка на модели платы (RTL SIMULATED, Verilator 5.046, конфигурация r8: хэш 93e29b18)

- `tools/r8_card_check.py --sim`:
  - **169 PASS, 0 FAIL**;
  - CAPS2 и CFG_HASH; все проверки одной операции, включая новые BFLY, CSWP и AFF;
  - многоинструкционные случаи r8.
- `tools/r8_kv_check.py --sim --prompt 9 --steps 2 --gen 3`:
  - **15 PASS, 0 FAIL**;
  - int8, kv6 и k6v6: логиты, `generate_card` и весь образ DRAM карта == ISA.
  - Позицию 261 покрывают RTL-тесты `tests/test_kv6_attention.py`.

Ещё исправлено: в `opchecks` проверки одной операции для BFLY и CSWP собирались в общей форме
VOP и были недопустимы (ISA отказывал). Теперь у BFLY, CSWP и AFF свои проверки (группа
`vpu-r8`).

## Открытые замечания (не относятся к плате r8)

- `tests/test_waits.py::test_waitc_rtl_hazard_stress[3]`: RTL != ISA в 44 словах TMEM.
  - Совпадает при WAITC 0 и 1, то есть дело не в wait-записях.
  - Было и до коммита 1f1db54.
  - Случайная программа общей Config(S=1) с GATHER при S=1; на плате хост GATHER при S=1 отклоняет.
  - Нужен отдельный разбор.
- `test_entries_are_kept` и `test_prebuild_keeps_the_sessions_entries` падают и на коммите до
  этих изменений (кэш программ в контейнере).
