<#
Registers two Windows scheduled tasks for paper_tracker.py. Safe to re-run: existing
tasks with the same names are replaced. Runs as the current user, only while logged on
(no stored password, no elevation). Paper only: these tasks read public data and write
paper_tracker.db / paper_tracker.log next to this script; nothing is ever placed.

  PolymarketPaperTracker-Cycle    every 2 hours   snapshot open pre-match tennis markets
  PolymarketPaperTracker-Resolve  daily at 04:15  record outcomes of settled matches

Missed runs (laptop asleep / off) execute when the machine is next available. The
machine is NOT woken to run them. Remove with:
  Unregister-ScheduledTask -TaskName PolymarketPaperTracker-Cycle  -Confirm:$false
  Unregister-ScheduledTask -TaskName PolymarketPaperTracker-Resolve -Confirm:$false
#>

$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python  = Join-Path $project ".venv\Scripts\python.exe"
$log     = Join-Path $project "paper_tracker.log"

if (-not (Test-Path $python)) { throw "venv python not found at $python" }

function New-TrackerAction([string]$command) {
    # Stamp each run in the log so a silent failure shows up as a missing/short entry.
    # Out-File -Encoding utf8 on both writes: `*>>` in Windows PowerShell 5.1 emits UTF-16
    # and produces an unreadable, ungreppable log when mixed with plain-text headers.
    $inner = "('=== ' + (Get-Date -Format s) + ' $command ===') | Out-File -FilePath '$log' -Append -Encoding utf8;" +
             " & '$python' paper_tracker.py $command 2>&1 | Out-File -FilePath '$log' -Append -Encoding utf8"
    New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -Command `"$inner`"" `
        -WorkingDirectory $project
}

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

$now = Get-Date
$nextHour = $now.Date.AddHours($now.Hour + 1)
$cycleTrigger = New-ScheduledTaskTrigger -Once -At $nextHour `
    -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 3650)
$resolveTrigger = New-ScheduledTaskTrigger -Daily -At "04:15"

Register-ScheduledTask -TaskName "PolymarketPaperTracker-Cycle" -Force `
    -Action (New-TrackerAction "cycle") -Trigger $cycleTrigger -Settings $settings `
    -Description "Snapshot open pre-match Polymarket tennis markets (paper only)." | Out-Null

Register-ScheduledTask -TaskName "PolymarketPaperTracker-Resolve" -Force `
    -Action (New-TrackerAction "resolve") -Trigger $resolveTrigger -Settings $settings `
    -Description "Record outcomes of settled Polymarket tennis markets (paper only)." | Out-Null

Get-ScheduledTask -TaskName "PolymarketPaperTracker-*" |
    ForEach-Object { $i = $_ | Get-ScheduledTaskInfo
        "{0}  state={1}  next run={2}" -f $_.TaskName, $_.State, $i.NextRunTime }
