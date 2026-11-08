# Vivado run properties for this build, sourced by build.tcl after it opens the project (the fmax
# tournament's whole-design component, otpu_impl, edits this file). One line per property:
#   set_property <PROPERTY> <value> [get_runs synth_1|impl_1]
# e.g. STEPS.SYNTH_DESIGN.ARGS.RETIMING true, STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt,
# STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore. Nothing else: no Tcl, no hooks, no
# timing exceptions (tools/tourney/gates.py checks it). Empty: Vivado's defaults.
set_property STRATEGY Performance_ExtraTimingOpt [get_runs impl_1]
# r8 LUT area (synthesis at 90.5 % LUT with Flow_PerfOptimized_high): that strategy's -no_lc,
# -resource_sharing off and -keep_equivalent_registers are reverted (LUT combining and shared
# arithmetic back; retiming stays); the core had +0.228 ns setup slack after synthesis.
set_property STEPS.SYNTH_DESIGN.ARGS.NO_LC false [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING auto [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS false [get_runs synth_1]
