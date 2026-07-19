# Install the SkillOpt-Sleep Cursor integration as a local Cursor plugin on Windows.
# Idempotent; prints what it does.

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CursorHome = if ($env:CURSOR_HOME) { $env:CURSOR_HOME } else { Join-Path $env:USERPROFILE ".cursor" }
$PluginDir = Join-Path $CursorHome "plugins\local\skillopt-sleep"
$SourceDir = Join-Path $RepoRoot "plugins\cursor"
$ManifestDir = Join-Path $PluginDir ".cursor-plugin"
$CommandDir = Join-Path $PluginDir "commands"
$SkillDir = Join-Path $PluginDir "skills\skillopt-sleep"

Write-Output "[install] repo: $RepoRoot"

New-Item -ItemType Directory -Path $ManifestDir -Force | Out-Null
New-Item -ItemType Directory -Path $CommandDir -Force | Out-Null
New-Item -ItemType Directory -Path $SkillDir -Force | Out-Null
Copy-Item (Join-Path $SourceDir ".cursor-plugin\plugin.json") (Join-Path $ManifestDir "plugin.json") -Force
Copy-Item (Join-Path $SourceDir "commands\skillopt-sleep.md") (Join-Path $CommandDir "skillopt-sleep.md") -Force
Copy-Item (Join-Path $SourceDir "skills\skillopt-sleep\SKILL.md") (Join-Path $SkillDir "SKILL.md") -Force
Copy-Item (Join-Path $SourceDir "README.md") (Join-Path $PluginDir "README.md") -Force
Copy-Item (Join-Path $SourceDir "LICENSE") (Join-Path $PluginDir "LICENSE") -Force

Write-Output "[install] plugin manifest -> $(Join-Path $ManifestDir 'plugin.json')"
Write-Output "[install] command         -> $(Join-Path $CommandDir 'skillopt-sleep.md')"
Write-Output "[install] skill           -> $(Join-Path $SkillDir 'SKILL.md')"
Write-Output ""
Write-Output "[install] Quit and reopen Cursor. The plugin should appear in Settings >"
Write-Output "Plugins under Installed."
Write-Output ""
Write-Output "For source-checkout runs, add this user environment variable:"
Write-Output "    [System.Environment]::SetEnvironmentVariable('SKILLOPT_SLEEP_REPO', '$RepoRoot', 'User')"
Write-Output ""
Write-Output "Alternatively, install a SkillOpt release that includes Cursor support so the"
Write-Output "skillopt-sleep command is on PATH."
Write-Output ""
Write-Output "Done. Try in Cursor:"
Write-Output "  /skillopt-sleep status"
