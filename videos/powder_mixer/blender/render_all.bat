@echo off
rem Renders the nine shots on the GPU, then assembles the film.
rem Safe to stop and start again: frames that already exist are skipped.
rem Edit the next line if Blender is installed somewhere else.
set BLENDER="C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"
cd /d "%~dp0"
for %%S in (Q1 Q2 Q3 Q4 S1 S2 F1 X6 X5) do %BLENDER% -b shot_%%S.blend -P use_gpu.py -a
%BLENDER% -b the_mix_off_MASTER.blend -a
echo Finished: the_mix_off_4K0001-5613.mp4
pause
