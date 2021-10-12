#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
coverage run --source=alignment -m pytest tests/test_alignment_holistic.py::test_alignment_holistic
coverage html
