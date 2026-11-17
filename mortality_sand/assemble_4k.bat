@echo off
rem Assemble the finished 4K film with ffmpeg (instead of Blender's Edit scene): the Cycles frames in render4k\,
rem the text layer (multiplied by overlay_matte.mp4, then overlay_fill.mp4 added), the last frame held for the
rem closing words (690 frames), and the narration. Put this file in the project folder and double-click it.
rem Needs ffmpeg:  winget install --id Gyan.FFmpeg -e   (then open a new window)

cd /d "%~dp0"
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found. Install it from PowerShell with:
  echo     winget install --id Gyan.FFmpeg -e
  echo then close this window and double-click this file again.
  pause
  exit /b 1
)
for /f %%n in ('dir /b render4k\frame_*.png ^| find /c /v ""') do set NFRAMES=%%n
echo Frames found in render4k: %NFRAMES%   (should be 8274)
if not exist overlay_matte.mp4 echo MISSING: overlay_matte.mp4
if not exist overlay_fill.mp4 echo MISSING: overlay_fill.mp4
if not exist narration.m4a echo MISSING: narration.m4a
echo.
echo Assembling two_thousand_years_of_catastrophe_4k.mp4 (about 10 to 40 minutes)...
ffmpeg -y -hide_banner -stats -loglevel error -framerate 30 -start_number 0 -i render4k\frame_%%04d.png -i overlay_matte.mp4 -i overlay_fill.mp4 -i narration.m4a -filter_complex "[0:v]tpad=stop_mode=clone:stop=690,format=gbrp[b];[1:v]format=gbrp[m];[2:v]format=gbrp[f];[b][m]blend=all_mode=multiply[bm];[bm][f]blend=all_mode=addition,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]" -map "[v]" -map 3:a -frames:v 8964 -c:v libx264 -preset medium -crf 17 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k -shortest -movflags +faststart two_thousand_years_of_catastrophe_4k.mp4
echo.
if errorlevel 1 (echo Something went wrong: copy the messages above and send them.) else (echo Done: two_thousand_years_of_catastrophe_4k.mp4)
pause
