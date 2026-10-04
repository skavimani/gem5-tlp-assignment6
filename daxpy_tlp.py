# daxpy_tlp.py - gem5 config for MSCS-531 Assignment 6 (TLP with MinorCPU)
#
# Multi-core shared-memory system in SE (syscall-emulation) mode:
#   N x MinorCPU (in-order)  -> private L1I/L1D per core
#   -> shared L2 (coherent crossbar, snooping MOESI) -> DDR3 memory
#
# The FloatSimd functional unit of every core is rebuilt from
# MinorDefaultFloatSimdFU (src/cpu/minor/BaseMinorCPU.py) with the
# opLat / issueLat given on the command line, so each design point can be
# run without editing and rebuilding gem5.
#
# Example:
#   build/X86/gem5.opt --outdir=m5out/test configs/assignment6/daxpy_tlp.py \
#       --isa x86 --num-cpus 4 --op-lat 3 --issue-lat 4 \
#       --binary ~/gem5-tlp/daxpy_mt --args "65536 4"

import argparse

import m5
from m5.objects import *

parser = argparse.ArgumentParser(description="Multi-core MinorCPU DAXPY")
parser.add_argument("--isa", choices=["x86", "riscv", "arm"], default="x86")
parser.add_argument("--binary", required=True, help="static binary to run")
parser.add_argument("--args", default="", help="arguments for the binary")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--op-lat", type=int, default=6, help="FloatSimd opLat")
parser.add_argument("--issue-lat", type=int, default=1,
                    help="FloatSimd issueLat")
parser.add_argument("--clock", default="2GHz")
parser.add_argument("--l1i-size", default="32KiB")
parser.add_argument("--l1d-size", default="32KiB")
parser.add_argument("--l2-size", default="1MiB")
parser.add_argument("--mem-size", default="2GiB")
args = parser.parse_args()

try:  # gem5 v22.1+ names the in-order CPU per ISA
    CPUClass = {"x86": "X86MinorCPU", "riscv": "RiscvMinorCPU",
                "arm": "ArmMinorCPU"}[args.isa]
    CPUClass = globals()[CPUClass]
except KeyError:  # older gem5 builds only have MinorCPU
    CPUClass = MinorCPU


# ---------------------------------------------------------------- caches
class L1ICache(Cache):
    size = args.l1i_size
    assoc = 2
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20


class L1DCache(Cache):
    size = args.l1d_size
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 8
    tgts_per_mshr = 20


class L2Cache(Cache):
    size = args.l2_size
    assoc = 8
    tag_latency = 10
    data_latency = 10
    response_latency = 10
    mshrs = 32
    tgts_per_mshr = 12


# ------------------------------------------- FloatSimd FU design point
def make_fu_pool(op_lat, issue_lat):
    """Same units, same order as MinorDefaultFUPool; only FloatSimd changes."""
    return MinorFUPool(funcUnits=[
        MinorDefaultIntFU(),
        MinorDefaultIntFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultFloatSimdFU(opLat=op_lat, issueLat=issue_lat),  # <- DSE
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),
    ])


# ---------------------------------------------------------------- system
system = System()
system.clk_domain = SrcClockDomain(clock=args.clock,
                                   voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(args.mem_size)]

system.cpu = [CPUClass(cpu_id=i) for i in range(args.num_cpus)]

system.membus = SystemXBar()
system.l2bus = L2XBar()

for cpu in system.cpu:
    cpu.executeFuncUnits = make_fu_pool(args.op_lat, args.issue_lat)

    cpu.icache = L1ICache()
    cpu.dcache = L1DCache()
    cpu.icache_port = cpu.icache.cpu_side
    cpu.dcache_port = cpu.dcache.cpu_side
    cpu.icache.mem_side = system.l2bus.cpu_side_ports
    cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    cpu.createInterruptController()
    if args.isa == "x86":
        cpu.interrupts[0].pio = system.membus.mem_side_ports
        cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
        cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.l2cache = L2Cache()
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# ---------------------------------------------------------------- workload
system.workload = SEWorkload.init_compatible(args.binary)
process = Process(pid=100)
process.cmd = [args.binary] + args.args.split()
for cpu in system.cpu:
    # every core shares ONE process: pthread_create() (clone) places each
    # new thread on the next idle core
    cpu.workload = process
    cpu.createThreads()

# ---------------------------------------------------------------- run
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Cores={args.num_cpus}  FloatSimd opLat={args.op_lat} "
      f"issueLat={args.issue_lat}  cmd={' '.join(process.cmd)}")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
