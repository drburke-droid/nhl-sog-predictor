@echo off
echo Syncing NHL database to GitHub...
cd /d C:\Users\rober\Desktop\API
git add nhl_data.db
git commit -m "Update database %date%"
git push
echo Done! Render will redeploy with fresh data.
pause
