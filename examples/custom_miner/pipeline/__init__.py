"""The sequential solve pipeline: register, solution, test kit, one fix.

Only the parts that need no model call live here so far -- the clock that
decides what may still be launched, the two comparison modes, and the
verification ladder. The stages that call a model are the layer above; every
one of them is a task with an eta, and `clock.decide` is the only launcher.
"""
