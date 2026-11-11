#!/bin/bash
# Rerun of two planted shapes of jobs/ruler_checks (specs/calib.md sections 1, 1b, 2), no pooling check.
# The job tarball carries jobs/ruler_checks as ruler_checks/ next to this run.sh; env JOB OUT FS DEADLINE.
SHAPES="randdir sae:0.001" POOLING=0 exec bash ruler_checks/run.sh
