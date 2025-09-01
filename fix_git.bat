@echo off
echo === Fixing Git Push Issue ===

echo.
echo 1. Removing large files from git tracking...
git rm --cached opencv-4.x.zip
git rm --cached -r opencv-4.x/

echo.
echo 2. Adding .gitignore changes...
git add .gitignore

echo.
echo 3. Adding other files...
git add manual_written_hit_detector.py
git add fix_git.py
git add yolov8n-pose.pt

echo.
echo 4. Committing changes...
git commit -m "Remove large files and update .gitignore"

echo.
echo 5. Attempting to push...
git push

echo.
echo === Done ===
pause
