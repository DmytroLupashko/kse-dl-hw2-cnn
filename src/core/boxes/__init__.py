"""Experimental Grad-CAM pseudo-boxes (models exp06-exp24), not used by later experiments.

`box`: `Box` geometry, `erasers` for Grad-CAM erase probes. `rows`: box files, `BoxExtras` (per-row boxes, backgrounds).
`ops`: `EraseBoxes`, `PasteBoxes`. `recipes`: box recipes. `pseudo`: CLI writing box files, `python -m core.boxes.pseudo`.
"""

from .box import Box, erasers
from .recipes import V3, AttentionDrop, BoxCrop, FaceErase, HeadErase, Paste
from .rows import BOXES, FACE_BOXES, HEAD_BOXES, read_boxes
