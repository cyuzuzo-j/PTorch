#!/bin/bash
echo "--- GIT ROOT ---"
git rev-parse --show-toplevel
echo "--- DATASET SIZE ---"
du -sh dataset
echo "--- LARGE FILES ---"
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort -rn | head -n 20
