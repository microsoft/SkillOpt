@echo off
setlocal enabledelayedexpansion

:: Resolve REPO_ROOT
set "SCRIPT_DIR=%~dp0"
:: Strip trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "REPO_ROOT="
if exist "%SCRIPT_DIR%\..\skillopt_sleep" (
    cd /d "%SCRIPT_DIR%\.."
    set "REPO_ROOT=%CD%"
    goto root_resolved
)
if not "%CLAUDE_PLUGIN_ROOT%"=="" if exist "%CLAUDE_PLUGIN_ROOT%\..\..\skillopt_sleep" (
    cd /d "%CLAUDE_PLUGIN_ROOT%\..\.."
    set "REPO_ROOT=%CD%"
    goto root_resolved
)
if not "%SKILLOPT_SLEEP_REPO%"=="" if exist "%SKILLOPT_SLEEP_REPO%\skillopt_sleep" (
    set "REPO_ROOT=%SKILLOPT_SLEEP_REPO%"
    goto root_resolved
)

:: Deliberately no search upward from the current directory: it is the
:: (possibly untrusted) project being worked on, and a skillopt_sleep folder
:: planted there would be run.

:root_resolved

if "%REPO_ROOT%"=="" goto fallback_mode

:: ── Source Checkout Mode ───────────────────────────────────────────
set "PY="
if not "%SKILLOPT_SLEEP_PYTHON%"=="" (
    set "PY=%SKILLOPT_SLEEP_PYTHON%"
    goto py_found
)

for %%p in (python3.exe python.exe py.exe) do (
    where %%p >nul 2>nul
    if !errorlevel! equ 0 (
        %%p -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if !errorlevel! equ 0 (
            set "PY=%%p"
            goto py_found
        )
    )
)

:py_found
if "%PY%"=="" (
    echo [sleep] ERROR: need Python >= 3.10 (found none). >&2
    exit /b 1
)

cd /d "%REPO_ROOT%"
if "%~1" == "" (
    "%PY%" -m skillopt_sleep status
) else (
    "%PY%" -m skillopt_sleep %*
)
exit /b !errorlevel!


:: ── Fallback Mode (No Source Checkout) ─────────────────────────────
:fallback_mode

:: Fallback 1: skillopt-sleep CLI on PATH (uv tool install / pipx / pip install).
where skillopt-sleep >nul 2>nul
if !errorlevel! neq 0 goto try_fallback_2

if "%~1" == "" (
    skillopt-sleep status
) else (
    skillopt-sleep %*
)
exit /b !errorlevel!

:try_fallback_2
:: Fallback 2: importable as a module (pip install into the active Python).
:: `python -c`/`-m` put the current directory (the user's project) first on
:: sys.path, so a planted skillopt_sleep.py there would run; strip it before
:: importing. The current directory itself is kept because the engine uses it
:: as the default project.
set "NO_CWD_IMPORT=import sys; sys.path[:] = [p for p in sys.path if p]; import skillopt_sleep"
set "NO_CWD_RUN=import sys; sys.path[:] = [p for p in sys.path if p]; import runpy; runpy.run_module('skillopt_sleep', run_name='__main__', alter_sys=True)"
set "PY="
for %%p in (python3.exe python.exe py.exe) do (
    where %%p >nul 2>nul
    if !errorlevel! equ 0 (
        %%p -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if !errorlevel! equ 0 (
            %%p -c "!NO_CWD_IMPORT!" >nul 2>nul
            if !errorlevel! equ 0 (
                set "PY=%%p"
                goto py_import_found
            )
        )
    )
)
:py_import_found

if "%PY%" == "" goto not_found

if "%~1" == "" (
    "%PY%" -c "!NO_CWD_RUN!" status
) else (
    "%PY%" -c "!NO_CWD_RUN!" %*
)
exit /b !errorlevel!

:not_found
echo [sleep] ERROR: could not locate the skillopt_sleep package. >&2
echo [sleep] Install it with 'uv tool install skillopt' or 'pip install skillopt', >&2
echo [sleep] or set SKILLOPT_SLEEP_REPO to a clone of the SkillOpt repo. >&2
exit /b 1
