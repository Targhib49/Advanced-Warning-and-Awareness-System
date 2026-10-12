# AWAS simulator bridge: one-shot setup for a Windows laptop with ETS2.
#
#   Right-click install.bat > Run as administrator   (admin only needed if ETS2
#   lives under C:\Program Files; otherwise a normal double-click is enough)
#
# What it does, all idempotent:
#   1. Python 3.12 for this user, if there is no Python 3.10+ (via winget)
#   2. A private venv in %LOCALAPPDATA%\awas-bridge with the bridge's packages
#   3. scs-sdk-plugin V.1.12.1 (checksum-pinned) into ETS2's plugins folder
#   4. "AWAS bridge" and "AWAS bridge (check)" shortcuts on the desktop

$ErrorActionPreference = "Stop"
$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Home_    = Join-Path $env:LOCALAPPDATA "awas-bridge"
$PluginUrl = "https://github.com/RenCloud/scs-sdk-plugin/releases/download/V.1.12.1/release_v_1_12_1.zip"
$PluginSha = "3679ec81407de0ebfb45241e496e2ad31e30a6e409ef0246fc86195bfee9c9a5"

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }

# ------------------------------------------------------------------ python
Say "Python"
function Find-Python {
    # Each candidate: the executable, then its own arguments as an array.
    $candidates = @(@("py", @("-3.12")), @("py", @("-3.11")), @("py", @("-3.10")), @("python", @()))
    foreach ($c in $candidates) {
        $exe, $pre = $c[0], $c[1]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $v = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($v -and [version]$v -ge [version]"3.10") { return ,@($exe, $pre) }
        } catch {}
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn "no Python 3.10+; installing Python 3.12 for this user"
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
        $py = Find-Python
    }
    if (-not $py) { throw "Install Python 3.12 from python.org (tick 'Add to PATH'), then run this again." }
}
$PyExe, $PyPre = $py[0], $py[1]
Write-Host "    using: $(& $PyExe @PyPre -V)"

# ------------------------------------------------------------------ venv
Say "Bridge environment in $Home_"
New-Item -ItemType Directory -Force -Path $Home_ | Out-Null
if (-not (Test-Path "$Home_\venv\Scripts\python.exe")) { & $PyExe @PyPre -m venv "$Home_\venv" }
& "$Home_\venv\Scripts\python.exe" -m pip install -q --upgrade pip
& "$Home_\venv\Scripts\python.exe" -m pip install -q -r "$Here\requirements.txt"
Copy-Item "$Here\awas_bridge.py" "$Home_\awas_bridge.py" -Force
Write-Host "    packages installed"

# ------------------------------------------------------------------ ETS2
Say "Euro Truck Simulator 2"
function Find-Ets2 {
    $steam = (Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -ErrorAction SilentlyContinue).SteamPath
    if (-not $steam) { return $null }
    $libs = @($steam)
    $vdf = Join-Path $steam "steamapps\libraryfolders.vdf"
    if (Test-Path $vdf) {
        foreach ($m in (Select-String -Path $vdf -Pattern '"path"\s+"([^"]+)"' -AllMatches).Matches) {
            $libs += $m.Groups[1].Value -replace '\\\\', '\'
        }
    }
    foreach ($lib in ($libs | Select-Object -Unique)) {
        $p = Join-Path $lib "steamapps\common\Euro Truck Simulator 2"
        if (Test-Path (Join-Path $p "bin\win_x64")) { return $p }
    }
    return $null
}
$ets2 = Find-Ets2
if (-not $ets2) {
    Warn "ETS2 not found through Steam. Install it, then run this again,"
    Warn "or copy Win64\scs-telemetry.dll into <ETS2>\bin\win_x64\plugins\ by hand."
} else {
    Write-Host "    found: $ets2"
    $plugins = Join-Path $ets2 "bin\win_x64\plugins"
    $dll = Join-Path $plugins "scs-telemetry.dll"
    if (Test-Path $dll) {
        Write-Host "    telemetry plugin already installed"
    } else {
        $zip = Join-Path $env:TEMP "scs-sdk-plugin-1.12.1.zip"
        Invoke-WebRequest -Uri $PluginUrl -OutFile $zip -UseBasicParsing
        $sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($sha -ne $PluginSha) { throw "plugin download checksum mismatch ($sha); not installing it" }
        $tmp = Join-Path $env:TEMP "scs-sdk-plugin-1.12.1"
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        New-Item -ItemType Directory -Force -Path $plugins | Out-Null
        try {
            Copy-Item (Join-Path $tmp "release_v_1_12_1\Win64\scs-telemetry.dll") $dll -Force
            Write-Host "    telemetry plugin installed (scs-sdk-plugin V.1.12.1, checksum verified)"
        } catch {
            Warn "could not write to $plugins - run install.bat as administrator."
            throw
        }
    }
}

# ------------------------------------------------------------------ shortcuts
Say "Desktop shortcuts"
$desk = [Environment]::GetFolderPath("Desktop")
$pyw = "$Home_\venv\Scripts\python.exe"
Set-Content -Path "$Home_\run.bat"   -Value "@echo off`r`n`"$pyw`" `"$Home_\awas_bridge.py`" %*`r`npause"
Set-Content -Path "$desk\AWAS bridge.bat"         -Value "@call `"$Home_\run.bat`""
Set-Content -Path "$desk\AWAS bridge (check).bat" -Value "@call `"$Home_\run.bat`" --check"
Write-Host "    'AWAS bridge' and 'AWAS bridge (check)' are on the desktop"

Say "Done"
Write-Host @"
    1. Start ETS2. It shows an 'SDK features' dialog the first time: press OK.
    2. Join the phone hotspot the Pi is on.
    3. Double-click 'AWAS bridge (check)' to see the speed, steering and lane.
       If 'awas.local' is not found, edit the shortcut to add  --pi <Pi's IP>.
"@
