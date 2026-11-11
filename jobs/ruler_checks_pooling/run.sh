#!/bin/bash
# Pooling check of jobs/ruler_checks only (no planted shapes): SHAPES=none. The job tarball carries
# jobs/ruler_checks as ruler_checks/ next to this run.sh; env JOB OUT FS DEADLINE as for the full job.
SHAPES=none exec bash ruler_checks/run.sh
