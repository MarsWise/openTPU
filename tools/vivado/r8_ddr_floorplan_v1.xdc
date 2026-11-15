# EXPERIMENT PLACEHOLDER — intentionally no active constraints.
# Do not add to a build as an effective floorplan: DDR PHY sites and candidate
# compute hierarchy/clock regions must first be identified in both routed DCPs.
# A soft pblock needs exact SLICE ranges and verified hierarchical membership;
# hardcoded or guessed coordinates could worsen timing and DDR stability.
# Preserve PHY, IODELAY, I/O, clocking, fixed DSP/BRAM and interface constraints.
# Proposed trial after measurement: select one compute hierarchy, create_pblock,
# resize_pblock with a measured generous SLICE range, add_cells_to_pblock, and
# set IS_SOFT TRUE; review overlap, CONTAIN_ROUTING and existing constraints.
# Rebuild, route and FPGA programming require separate user authorization.
