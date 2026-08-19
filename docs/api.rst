API Reference
=============

This reference is generated from the in-source docstrings.

Problem classes
---------------

.. automodule:: asymptotics.core.problem
   :members: PerturbationEquation, AlgebraicEquation, ODE, AlgebraicSystem
   :member-order: bysource

.. automodule:: asymptotics.core.ode_system
   :members: ODESystem
   :member-order: bysource

Expansion hierarchies
---------------------

Regular perturbation (ODE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.regular_ode
   :members:
   :member-order: bysource

.. automodule:: asymptotics.core.hierarchy
   :members:
   :member-order: bysource

Lindstedt–Poincaré
^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.lindstedt
   :members:
   :member-order: bysource

Method of multiple scales
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.multiple_scales
   :members:
   :member-order: bysource

Matched asymptotics (boundary layers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.boundary_layer
   :members:
   :member-order: bysource

Coupled ODE systems
^^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.regular_ode_system
   :members:
   :member-order: bysource

.. automodule:: asymptotics.core.system_hierarchy
   :members:
   :member-order: bysource

Step-by-step expansion
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: asymptotics.methods.stepwise
   :members:
   :member-order: bysource

Numerics, evaluation, and export
--------------------------------

.. automodule:: asymptotics.numerics
   :members: compare_numeric, error_norms
   :member-order: bysource

.. automodule:: asymptotics.eval
   :members: eval_hierarchy
   :member-order: bysource

.. automodule:: asymptotics.latex_export
   :members: to_latex
   :member-order: bysource

Exceptions
----------

.. automodule:: asymptotics.core.exceptions
   :members:
   :member-order: bysource

.. autoclass:: asymptotics.core.conditions.ConditionError
   :members:
