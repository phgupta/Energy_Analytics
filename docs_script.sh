#!/usr/bin/env bash

# Build docs
cd docs/sphinx/
mkdir _static # Gives error if _static is not present
make clean
make html

# Delete previous html files in docs/
cd ..
ls -a | grep -v "sphinx" ".nojekyll" | xargs rm -rf

# Copy updated html files to docs/
cp -r sphinx/build/html/* .

# Check if docs is updated
# open index.html
