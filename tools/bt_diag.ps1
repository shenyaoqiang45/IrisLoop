param([int]$Seconds = 10)

$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime]

$global:ev = @{ Added = 0; Updated = 0; Removed = 0; Completed = 0; Stopped = 0 }
$global:names = @()

$sel = [Windows.Devices.Bluetooth.BluetoothDevice]::GetDeviceSelector()
Write-Output ("SELECTOR: {0}" -f $sel.Substring(0, [Math]::Min(160, $sel.Length)))

$watcher = [Windows.Devices.Enumeration.DeviceInformation]::CreateWatcher($sel)

Register-ObjectEvent -InputObject $watcher -EventName Added -Action {
    $global:ev.Added++
    $global:names += ("ADD  " + $EventArgs.Name + " | " + $EventArgs.Id)
} | Out-Null

Register-ObjectEvent -InputObject $watcher -EventName Updated -Action {
    $global:ev.Updated++
} | Out-Null

Register-ObjectEvent -InputObject $watcher -EventName EnumerationCompleted -Action {
    $global:ev.Completed++
} | Out-Null

Register-ObjectEvent -InputObject $watcher -EventName Stopped -Action {
    $global:ev.Stopped++
} | Out-Null

$watcher.Start()
Write-Output ("watcher status after Start: {0}" -f $watcher.Status)
for ($i = 0; $i -lt $Seconds; $i++) { Start-Sleep -Seconds 1 }
try { $watcher.Stop() } catch {}

Write-Output "--- EVENT COUNTS ---"
Write-Output ("Added={0} Updated={1} Completed={2} Stopped={3}" -f $global:ev.Added, $global:ev.Updated, $global:ev.Completed, $global:ev.Stopped)
Write-Output "--- DEVICES ---"
if ($global:names.Count -eq 0) { Write-Output "(none)" } else { $global:names | ForEach-Object { Write-Output ("  " + $_) } }
