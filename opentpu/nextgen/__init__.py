"""Shared r7 planning and strict execution utilities.

This package is intentionally independent from FPGA RTL. Importing it does not
claim hardware support for any new model or operator.
"""
from .planner import (DDR3_BYTES, Buffer, Allocation, MemoryPlan, MemoryBudgetError,
                      plan_buffers, Stage, ExecutionTrace, HostComputeError)

__all__ = ["DDR3_BYTES", "Buffer", "Allocation", "MemoryPlan", "MemoryBudgetError",
           "plan_buffers", "Stage", "ExecutionTrace", "HostComputeError"]
