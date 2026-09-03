@echo off
REM ---------------------------------------------------------------
REM  Module 03 - React dashboard (Create React App, no Vite)
REM ---------------------------------------------------------------
cd /d %~dp0
if not exist node_modules (
  echo Installing front-end dependencies, this runs once...
  call npm install
)
echo Starting the dashboard on http://localhost:3000
call npm start
