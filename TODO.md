# CubeSat Error Fix TODO

## Steps from approved plan:
- [x] Step 1: Install missing dependency 'simple-pid' (verified import success)
- [x] Step 2: Install pygame & numpy (pip install failed due to pygame build error on Windows/Python 3.14; numpy ok)
- [ ] Step 3: Install Microsoft C++ Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/), restart terminal/VSCode, retry 'pip install pygame'
- [ ] Step 4: Verify deps: python -c "import pygame, numpy, simple_pid"
- [ ] Step 5: Test run 'python system.py'
- [ ] Step 6: Confirm no code errors, complete task
