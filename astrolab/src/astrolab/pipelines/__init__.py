"""Declarative pipelines composing core, instruments, and science modules.

A pipeline's job is orchestration and bookkeeping: run the stages in order, thread the
provenance through, and refuse to write a result the data do not support. It contains no
science of its own -- if a pipeline is doing arithmetic on measurements, that arithmetic
belongs in a science module where it can be tested.
"""
