# Claude: load the r8 bitstream (a4138552, CFG_HASH 93e29b18) into the FPGA over JTAG, VOLATILE
# (SRAM only, no flash: a power cycle brings back the flash image). Run only by r8_load.cmd,
# which the user starts after deciding to load. After it: r8_restore_pcie.cmd (PCIe rescan on
# the host), then r8_card_tests.cmd.
if {$argc != 1} { error "usage: vivado -mode batch -source r8_load.tcl -tclargs <otpu.bit>" }
set bitfile [lindex $argv 0]
if {![file exists $bitfile]} { error "r8 bitstream not found: $bitfile" }
open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets]
puts "JTAG targets: $targets"
if {[llength $targets] != 1} { error "expected exactly one JTAG target" }
open_hw_target [lindex $targets 0]
set devices [get_hw_devices]
if {[llength $devices] != 1} { error "expected exactly one JTAG device" }
set device [lindex $devices 0]
set part [get_property PART $device]
if {$part ne "xc7k480t"} { error "unexpected JTAG part: $part" }
current_hw_device $device
set_property PROGRAM.FILE $bitfile $device
program_hw_devices $device
refresh_hw_device $device
set done [get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $device]
puts "PROGRAMMED_DEVICE name=[get_property NAME $device] part=$part DONE=$done file=$bitfile"
if {$done != 1} { error "FPGA DONE pin not asserted" }
close_hw_manager
exit
