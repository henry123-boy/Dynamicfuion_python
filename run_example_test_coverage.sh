#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
coverage run --source=alignment.deform_net -m pytest tests/test_alignment_holistic.py::test_alignment_holistic2
coverage html
