Architecture
============

`pymead` is based on a two-layer architecture: the top-level GUI layer and the underlying API layer. As a general
rule, any user interaction with the GUI layer (pressing buttons, dragging objects, modifying values, etc.)
sends a signal first to the core API layer to determine what the result of the interaction should be. After
varying levels of logic are applied, the result is sent back to the GUI layer, often to both the tree and canvas
sub-layers. This software design leads to simpler, more linear code, and avoids duplicating code in both the tree
and canvas sub-layers. The general layout of the architecture is shown in the figure below.

.. figure:: images/pymead-diagram_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   API-GUI relationship in pymead

.. figure:: images/pymead-diagram_light.*
   :width: 600px
   :align: center
   :class: only-light

   API-GUI relationship in pymead
