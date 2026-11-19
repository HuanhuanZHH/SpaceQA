param(
    [string]$Server = 'user@server',
    [string]$RemoteRepository = '/absolute/path/to/spaceqa-agent',
    [int]$LocalPort = 18765,
    [int]$RemotePort = 8765,
    [string]$BindAddress = ''
)
$ErrorActionPreference = 'Stop'
$Url = "http://127.0.0.1:$LocalPort"
$Existing = $null
try { $Existing = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2 } catch { }
if ($Existing -and $Existing.service -eq 'spaceqa-web') {
    Write-Host "SpaceQA is already available: $Url"
    return
}
if ($RemoteRepository -notmatch '^/[A-Za-z0-9_./-]+$') {
    throw 'RemoteRepository must be an absolute server path without shell metacharacters.'
}
$SshArguments = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')
if ($BindAddress) { $SshArguments += @('-b', $BindAddress) }
& ssh @SshArguments $Server "bash $RemoteRepository/deploy/web_service/start_server.sh start"
if ($LASTEXITCODE -ne 0) { throw 'The server could not be started. Check SSH connectivity and the server log.' }
Write-Host "Open http://127.0.0.1:$LocalPort in your browser."
Write-Host 'Keep this window open. Press Ctrl+C to close the browser connection.'
& ssh @SshArguments -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    -L "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}" $Server
if ($LASTEXITCODE -ne 0) { throw 'SSH forwarding ended. Check network access and whether the local port is in use.' }
