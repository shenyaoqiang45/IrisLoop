param([int]$Seconds = 15)

$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime]
[void][Windows.Devices.Bluetooth.BluetoothDevice,Windows.Devices.Bluetooth,ContentType=WindowsRuntime]
[void][Windows.Devices.Bluetooth.BluetoothAdapter,Windows.Devices.Bluetooth,ContentType=WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and
    $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($op, $type) {
    $m = $asTaskGeneric.MakeGenericMethod($type)
    $t = $m.Invoke($null, @($op))
    $t.Wait(-1) | Out-Null
    return $t.Result
}

Write-Output "=== BLUETOOTH ADAPTER ==="
try {
    $ad = Await ([Windows.Devices.Bluetooth.BluetoothAdapter]::GetDefaultAsync()) ([Windows.Devices.Bluetooth.BluetoothAdapter])
    if ($ad) {
        Write-Output ("  address      : {0:X12}" -f $ad.BluetoothAddress)
        Write-Output ("  classic(BR)  : {0}" -f $ad.IsClassicSupported)
        Write-Output ("  lowEnergy    : {0}" -f $ad.IsLowEnergySupported)
        Write-Output ("  centralRole  : {0}" -f $ad.IsCentralRoleSupported)
        Write-Output ("  peripheralRole: {0}" -f $ad.IsPeripheralRoleSupported)
    } else { Write-Output "  NO DEFAULT ADAPTER" }
} catch { Write-Output ("  adapter error: {0}" -f $_.Exception.Message) }

Write-Output ""
Write-Output "=== ALREADY PAIRED ==="
try {
    $selP = [Windows.Devices.Bluetooth.BluetoothDevice]::GetDeviceSelectorFromPairingState($true)
    $list = Await ([Windows.Devices.Enumeration.DeviceInformation]::FindAllAsync($selP)) ([Windows.Devices.Enumeration.DeviceInformationCollection])
    $n = 0; if ($list) { $n = $list.Count }
    Write-Output ("  count: {0}" -f $n)
    if ($list) { foreach ($d in $list) { Write-Output ("  - {0}  [{1}]" -f $d.Name, $d.Id) } }
} catch { Write-Output ("  paired error: {0}" -f $_.Exception.Message) }

Write-Output ""
Write-Output ("=== ACTIVE DISCOVERY ({0}s) ===" -f $Seconds)
Write-Output "  scanning..."

$global:btFound = @{}
$sel = [Windows.Devices.Bluetooth.BluetoothDevice]::GetDeviceSelector()
$watcher = [Windows.Devices.Enumeration.DeviceInformation]::CreateWatcher($sel)

$collect = {
    try {
        $di = $EventArgs
        if (-not $di) { return }
        $id = [string]$di.Id
        $addr = ''
        if ($id -match '([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})$') {
            $addr = $Matches[1].ToUpper()
        }
        $kind = if ($id -like 'BluetoothLE#*') { 'LE' } else { 'Classic' }
        $rssi = ''
        try { if ($di.Properties -and $di.Properties.ContainsKey('System.Devices.Aep.SignalStrength')) {
            $rssi = [string]$di.Properties['System.Devices.Aep.SignalStrength'] } } catch {}
        $paired = $false
        try { $paired = [bool]$di.Pairing.IsPaired } catch {}
        $name = [string]$di.Name
        if (-not $name) { $name = '(unnamed)' }
        $global:btFound[$id] = [pscustomobject]@{
            Name = $name; Address = $addr; Kind = $kind; RSSI = $rssi; Paired = $paired
        }
    } catch {}
}

Register-ObjectEvent -InputObject $watcher -EventName Added   -Action $collect | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Updated -Action $collect | Out-Null

$watcher.Start()
for ($i = 0; $i -lt $Seconds; $i++) { Start-Sleep -Seconds 1 }
try { $watcher.Stop() } catch {}

$items = $global:btFound.Values | Sort-Object Kind, Name
Write-Output ("  found: {0}" -f @($items).Count)
Write-Output ""
$items | ForEach-Object {
    $flag = if ($_.Paired) { 'PAIRED' } else { '       ' }
    Write-Output ("  [{0}] {1,-8} {2,-20} rssi={3,-5} {4}" -f $_.Kind, $_.Address, $_.Name, $_.RSSI, $flag)
}
Write-Output ""
Write-Output "  (Kind: Classic=BR/EDR, LE=Low Energy; PAIRED=already paired)"
