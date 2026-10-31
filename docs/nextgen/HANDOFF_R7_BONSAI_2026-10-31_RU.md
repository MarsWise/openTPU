# openTPU r7 — состояние и продолжение, 31 октября 2026

Этот файл — точка входа для продолжения работы после этого чата. Он отделяет
реализованный код, моделирование и физические измерения от незавершённых задач.

## Приоритет и ограничения пользователя

- Текущий приоритет: **только ускорение Bonsai 2 4B на Kintex-7**. Работы по
  YOLO, VLA, Diffusion и другим моделям отложены до измерения оптимизации Bonsai.
- Измерить полный путь генерации, DDR, повторные запросы GSC, MXU, циклы и
  host overhead; затем реализовать и проверить повторное использование весов.
- Сохранять стабильную r5; работать в `MarsWise/openTPU`, `feature/r7-fpga`,
  не менять upstream `FeSens/openTPU`, не трогать его draft PR, не force-push.
- Не программировать FPGA без отдельного явного разрешения; QSPI/flash требует
  отдельного разрешения. Пользователь разрешил базовый профиль на **уже
  загруженной r5**, что не является разрешением на новую прошивку.
- Результат ускорения допустимо заявлять только после физического сравнения
  до/после с одним checkpoint, prompt, частотой, конфигурацией и квантованием.
- Общая долгосрочная цель ТЗ: одна прошивка r7 для FPGA-only **численного**
  inference. CPU может обслуживать токенизацию, I/O и запуск, но не численные
  операции модели. Эта цель пока не выполнена.

## Среда и версии

| Объект | Значение |
|---|---|
| Рабочая копия | `C:\myOpenTPU\openTPU-r7` |
| Репозиторий | `https://github.com/MarsWise/openTPU`, ветка `feature/r7-fpga` |
| Последний кодовый коммит на момент составления | `95124c98` — эксперимент разбиения GSC; затем добавлен этот handoff. Проверять фактический `git log` и `git status`. |
| Исходная физическая r5 | RTL `45b1c1d3`; отчётный коммит `55b975c1` не является другим bitstream |
| r6 | `20a49288`; объединена в r7 коммитом `9b399c51` |
| Плата | YPCB-00338, Kintex-7 XC7K480T-2, 4 ГиБ DDR3, PCIe Gen2 x8 |
| Сервер | `nvx@192.168.0.157`; SSH-ключ работал при последней проверке |
| Модель на сервере | `/home/nvx/openTPU_BitNET/models/Ternary-Bonsai-2-4B-mlx-2bit` |
| Изолированная копия для профиля | `/home/nvx/bonsai-profile-9dde9067-Tal5gvg4` (архив исходников `9dde9067` + скрипт запуска) |
| Python сервера | `/home/nvx/hfref/bin/python` |
| Vivado | `C:\Xilinx\Vivado\2024.2\bin\vivado.bat`, версия 2024.2.2 |

Не отслеживаемая папка `C:\myOpenTPU\openTPU-r7\.venv` принадлежит
пользователю; не удалять и не добавлять в коммиты. На плате при последней
проверке находилась r5 `BUILD_ID=45b1c1d3`, `CFG_HASH=a5bcbaef`, частота
133333 kHz, обе DDR калиброваны. Микротесты записывали данные в DDR; перед
новым запуском модели её образ потребуется загрузить заново. FPGA и flash в
этом измерении не программировались.

## Что реально реализовано

1. **База r5:** DDR3/PCIe/XDMA, MXU/VPU, TMM, TFN, PAIR/QUAD, native ternary
   GSC и ограниченные физические тесты. Отчёт: [R5 board validation](R5_45B1_BOARD_VALIDATION_2026-10-30.md).
2. **r6 в r7:** CMP/SEL, отдельный BIN и BIN+GSC, VU8, отдельные image/NMS
   примитивы, строгий greedy-путь vLLM. Есть ISA/RTL проверки отдельных частей;
   они не подтверждены вместе в новом физическом bitstream.
3. **Bonsai 2 4B:** загрузчик native ternary/Hadamard, преобразование Hadamard
   на VPU, TMM GSC с отдельным FP32 масштабом на группу. На r5 один prompt
   дал logits cosine `0.999926239` против CPU и совпавший следующий токен.
   Это ограниченная проверка, не длинная генерация и не FPGA-only.
4. **r7 память:** lifetime planner подключён к реальному выделению активаций
   `cnngraph.build(..., reuse_memory=True)`; маленькие и residual-графы имеют
   ISA тесты. Память всех моделей и общий resident executor не реализованы.
5. **Строгий режим:** новые `--strict-fpga-only` отказывают в запуске гибридных
   YOLO/Diffusion/SmolVLA/chat путей. Это fail-closed защита, не перенос их
   вычислений. Старые флаги `--strict-fpga` не доказывают FPGA-only.
6. **Отдельные эксперименты:** DDPM epsilon step прошёл CPU/ISA/RTL проверки;
   Philox4x32-10 прошёл отдельный RTL-тест. Ни DDPM, ни Philox не подключены
   к полным моделям; Philox находится в `rtl/exp/philox` и не входит в текущую
   прошивку. Эти направления сейчас на паузе.

## Физический baseline Bonsai

Полный отчёт и исходные JSON: [BONSAI_R5_BASELINE_2026-10-31.md](BONSAI_R5_BASELINE_2026-10-31.md).
Данные также сохранены локально в `C:\myOpenTPU\reports\bonsai-performance-2026-10-31`.

- Один физический запуск: prompt `Write a short story.`, 17 prompt-токенов,
  greedy, cap 512, seed 0. Первый output token выбирал CPU; затем 31 токен
  сгенерирован за один запуск на устройстве.
- Decode: **0,6805 токена/с wall**, **0,7061 токена/с по FPGA-циклам**;
  188,833 млн циклов/токен. Prefill 27,21 с, полный turn 72,76 с без загрузки
  модели; compile bucket около 1,65 с.
- `MXU_BUSY/RUNNING` 99,76%, `MXU_MAC/RUNNING` 17,41%,
  `MXU_STARVE/RUNNING` 17,52%. Последний счётчик считает пустой FIFO, но не
  всё ожидание частично заполненной строки.
- `DRAM_RD × 64` около 17,896 ГБ/токен — **все** чтения в интервале, не только
  веса или физические DDR pins. Port B — 33,35 млн запросов/токен, тоже не
  только GSC. ECC 0/0 на контрольных границах; это не длительная проверка.
- Микробенч `N=2560, KB=20` на r5, синтетические одинаковые масштабы: RSC
  12 800 запросов port B / 51 247 циклов после приблизительного baseline;
  GSC 51 200 запросов / 349 185 циклов. Оба результата побитно совпали с ISA,
  ECC 0/0. Разные реальные групповые масштабы **нельзя** заменить на RSC.
- На этом основании GSC rereads и последовательный проход — подтверждённый
  кандидат, но точную долю причин 17,896 ГБ/токен и предстоящий выигрыш нужно
  установить; не считать отношение микробенчмарка ускорением всей модели.

## Текущая работа над GSC

Исходный `rtl/mxu/otpu_mxu.sv`, `otpu_tmm_gsc`, выполняет все N выходных строк
для группы k и затем следующую группу; packed chunk повторно считывается для
четырёх ternary групп. Каждая группа должна отдельно масштабироваться до
FP32-накопления. Обычный PAIR/QUAD складывает целые суммы до общего масштаба,
поэтому напрямую включать его для GSC с разными масштабами неверно.

Добавлен **только эксперимент** `opentpu/nextgen/gsc_tiling.py`: разбиение TMM
по выходным столбцам, без изменения порядка K для одного результата. Он не
подключён к компилятору, отклоняет RMAX и не содержит кэш или повторное
использование весов. Само разбиение не уменьшает трафик и не является
оптимизацией. Тесты: 17 ISA случаев, 2 RTL с AXI stalls и WAITC; все прошли.
Предыдущая GSC-регрессия `tests/test_tmm_gsc.py` — 14 passed; отдельный RTL
Hadamard+GSC — 1 passed. Изменение RTL для reuse **ещё не реализовано**.

Следующему исполнителю: спроектировать небольшой tile-local replay уже
прочитанного packed chunk с учётом размера существующего FIFO и ресурсов.
Рассмотреть сохранение одного тайла между группами или переупорядочивание,
но сначала доказать: адреса `w1/w2/w3`, раздельные масштабы, FP32 порядок,
ACC, RMAX (fallback или корректная глобальная редукция), хвосты N/KB, M,
ACT ranges, готовность/задержки ответов, stall/reset, одновременную DMA-запись
и счётчики/scoreboard. Сохранять неизменный исходный путь для неподдержанных
случаев. Проверить независимый float, bit-exact ISA и RTL с задержками, затем
синтез/route и только после разрешения — физическое сравнение с baseline.

## Vivado: текущее состояние

`tools/build_r7_133.ps1` и `start_r7_133.cmd` готовят эксперимент r7 m8 на
133,33 МГц, TGSC+VCMP+TBIN+TBGSC+VU8; FPGA не программируют. Исходный проект:
`C:\myOpenTPU\openTPU-r7\build\vivado-r7-m8-133.33-09e9d34f-native\otpu.xpr`.
Его аппаратный исходный commit — **`09e9d34f`**, не текущий HEAD; более поздние
экспериментальные файлы не вошли в этот дизайн.

- Synthesis прошёл, включая T4 mapping: 8 BRAM, 0 FF; защёлок и неохваченных
  внутренних тактовых endpoints по gate не обнаружено.
- Исходный implementation завершился `EXCEPTION_ACCESS_VIOLATION` внутри
  Vivado `phys_opt`; полного route/bitstream в этом запуске нет.
- Независимый replay из `otpu.runs/impl_1/otpu_fpga_top_ld_opt.dcp`, без новых
  pblocks, выполнил placement и `phys_opt -directive Explore` и сохранил
  `C:\myOpenTPU\openTPU-r7\build\bonsai-r7-place-replay-09e9d34f\physopt\design.dcp`.
- Отдельный **диагностический route** стартовал из этого checkpoint:
  `C:\myOpenTPU\openTPU-r7\build\bonsai-r7-route-replay-09e9d34f\replay.log`.
  Последний видимый этап — инициализация `route_design -directive NoTimingRelaxation`.
  Проверить актуальный лог/процесс: этот replay не создаёт bitstream и не
  включает все финальные signoff/ECO шаги. До полного route и проверки
  setup/hold, DRC, CDC и портовых ограничений timing не считается закрытым.

## Команды и проверки

Программные тесты (Windows Python 3.11):

```powershell
cd C:\myOpenTPU\openTPU-r7
& 'C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe' -m pytest --noconftest tests/test_bonsai2.py tests/test_bonsai_gsc_bench.py tests/test_bonsai_profile.py -q -k 'not rtl'
& 'C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe' -m pytest --noconftest tests/test_bonsai_gsc_tiling.py -q -k 'not rtl'
```

RTL тесты выполнялись в Docker `otpu-sim-prepared:r6-local`, Verilator cache
`C:/myOpenTPU/r6-dev/build/verilator`; `tests/test_tmm_gsc.py` и
`tests/test_bonsai_gsc_tiling.py` прошли с pytest/conftest внутри контейнера.

Базовый запуск на сервере уже выполнен. Для **нового** замера нужно обеспечить
те же параметры и уникальный файл отчёта; стандартный r5-скрипт не перезапишет
существующий JSON:

```bash
BONSAI_REPORT=/новый/путь/bonsai-r5-before.json bash /home/nvx/bonsai-profile-9dde9067-Tal5gvg4/tools/bonsai_profile_r5.sh
```

Запуск меняет DRAM и выполняет модель; он не программирует FPGA. Скрипт требует
ожидаемые build/hash, TGSC, DDR calibration, инициализированную память, чистый
ECC и свободную карту. Новую прошивку без отдельного разрешения не загружать.
`tools/compare_bonsai_profiles.py before.json after.json` отказывается сравнивать
разные хеши модели/prompt, параметры, частоту и выдачу; отдельно нужны сравнение
logits, ECC и timing.

## Отложенные требования исходного ТЗ

После измерения/оптимизации Bonsai вернуться к полной матрице операторов и
FPGA-only: device resize и подключение к YOLO/VLM/VLA; полный SmolVLA граф и
action loop; полный Diffusion PushT graph/scheduler/noise; vLLM sampling,
logprobs, мультислоты; SD1.5, Kokoro-82M и MusicGen Small. Нужны общий resident
graph executor, device-side loops и аудит фактических CPU fallback, полная
оценка DDR ресурсов и end-to-end реальные модели. Текущая матрица —
[R7_OPERATOR_COVERAGE.md](R7_OPERATOR_COVERAGE.md); она не является полным
автоматически снятым графом всех checkpoint.
