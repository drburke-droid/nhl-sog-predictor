@echo off
setlocal
rem 2,000 years of catastrophe: check every frame, render any that are missing or broken, then build the 4K film.
rem Double-click it. It finds the project folder (the one with render4k) next to itself, one level up, or in Downloads.

echo ================================================================
echo  2,000 years of catastrophe: fix the frames, then build the film
echo ================================================================
set "HERE=%~dp0"
cd /d "%HERE%"
if not exist "render4k\" if exist "..\render4k\" cd ..
if not exist "render4k\" if exist "%USERPROFILE%\Downloads\two_thousand_years_cycles_4k\render4k\" cd /d "%USERPROFILE%\Downloads\two_thousand_years_cycles_4k"
if not exist "render4k\" (
  echo Could not find the project folder ^(the one that contains render4k^).
  echo Put this file in that folder, for example C:\Users\%USERNAME%\Downloads\two_thousand_years_cycles_4k
  pause
  exit /b 1
)
echo Project folder: %CD%

set "MISSING="
for %%F in (two_thousand_years_of_catastrophe.blend overlay_fill.mp4 overlay_matte.mp4 narration.m4a) do if not exist "%%F" (echo MISSING: %%F & set "MISSING=1")
if not exist "cache\static.npz" (echo MISSING: the cache folder & set "MISSING=1")
if not exist "%HERE%fix_frames.py" (echo MISSING: fix_frames.py, which must be next to this file & set "MISSING=1")
if defined MISSING (
  echo Those files must be in %CD%. Nothing was changed.
  pause
  exit /b 1
)

rem ---- Blender: on the PATH, in Program Files (the newest version), or Steam; else ask
set "BLENDER="
for /f "delims=" %%B in ('where blender 2^>nul') do if not defined BLENDER set "BLENDER=%%B"
if not defined BLENDER for /d %%D in ("%ProgramFiles%\Blender Foundation\Blender*") do if exist "%%D\blender.exe" set "BLENDER=%%D\blender.exe"
if not defined BLENDER if exist "%ProgramFiles(x86)%\Steam\steamapps\common\Blender\blender.exe" set "BLENDER=%ProgramFiles(x86)%\Steam\steamapps\common\Blender\blender.exe"
if not defined BLENDER (
  echo Could not find Blender. Paste the full path to blender.exe and press Enter:
  set /p "BLENDER="
)
if defined BLENDER set "BLENDER=%BLENDER:"=%"
if not exist "%BLENDER%" (
  echo Blender was not found at "%BLENDER%".
  pause
  exit /b 1
)
echo Blender: %BLENDER%

rem ---- ffmpeg: checked now, before the long render
set "FFMPEG="
for /f "delims=" %%G in ('where ffmpeg 2^>nul') do if not defined FFMPEG set "FFMPEG=%%G"
if not defined FFMPEG if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
if not defined FFMPEG (
  echo ffmpeg is not installed. Installing it now with winget...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
)
if not defined FFMPEG (
  echo Could not install ffmpeg. In PowerShell run:  winget install --id Gyan.FFmpeg -e
  echo then double-click this file again.
  pause
  exit /b 1
)
echo ffmpeg: %FFMPEG%

echo.
echo ---------------------------------------------------------------
echo  Step 1 of 2: check all 8,274 frames; render only the missing or
echo  broken ones, on your graphics card. Frames already there are
echo  skipped. If each frame takes many minutes, Blender is using the
echo  processor: open Blender, Edit ^> Preferences ^> System ^> Cycles
echo  Render Devices, choose OptiX or CUDA, tick your card, close
echo  Blender and run this again. You can stop at any time (close the
echo  window) and run it again: it carries on where it stopped.
echo ---------------------------------------------------------------
"%BLENDER%" -b -y two_thousand_years_of_catastrophe.blend -S Film --python-exit-code 1 -P "%HERE%fix_frames.py"
if errorlevel 1 (
  echo.
  echo Some frames are still missing or broken: see the messages above. The film was not built.
  pause
  exit /b 1
)

echo.
echo ---------------------------------------------------------------
echo  Step 2 of 2: building two_thousand_years_of_catastrophe_4k.mp4
echo  (frames + text + bar race + held ending + narration), about
echo  10 to 40 minutes
echo ---------------------------------------------------------------
"%FFMPEG%" -y -hide_banner -stats -loglevel error -framerate 30 -start_number 0 -i render4k\frame_%%04d.png -i overlay_matte.mp4 -i overlay_fill.mp4 -i narration.m4a -filter_complex "[0:v]tpad=stop_mode=clone:stop=690,format=gbrp[b];[1:v]format=gbrp[m];[2:v]format=gbrp[f];[b][m]blend=all_mode=multiply[bm];[bm][f]blend=all_mode=addition,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]" -map "[v]" -map 3:a -frames:v 8964 -c:v libx264 -preset medium -crf 17 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k -shortest -movflags +faststart two_thousand_years_of_catastrophe_4k.mp4
if errorlevel 1 (
  echo.
  echo ffmpeg reported an error: copy the messages above and send them.
  pause
  exit /b 1
)
echo.
echo ================================================================
echo  Done: %CD%\two_thousand_years_of_catastrophe_4k.mp4
echo  Play it through (the sand should keep moving to the end), then
echo  upload it as a new video.
echo ================================================================
pause
