## This document describes the "python" branches

| Branch Name | Description |
| :---------: | :-----------|
| `python` | This is the "main" python branch. It has working versions of all the homeworks up to Lesson 10. There's an abandoned Lesson 11 as well. The first ten can be referenced if you need hints. 
| `python_start_here` | This is the "starting" point with the infrastructure similar to what the C++ starts with; things that handle framebuffers, etc. More info at the root-level `README.md`.
| `python_numba` | This was a WIP not-cleaned-up attempt to port to `numba` to try to make everything faster. The goal was to get `lesson_10_homework.py` to run, since Lesson 11 required 1000 new renders and Lesson 10 was already ~35s to render _once_ on my laptop at a lower resolution. I was also trying to learn `numba` at the time, but saw that the wrappers needed to share the pixel data back and forth between C and Python was going to kill most of the progress that was being made. Self-tests fail miserably, but I may revisit some day (doubtful).
