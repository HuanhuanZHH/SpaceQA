param(
    [string]$Server = 'tangjiarui@222.195.72.220',
    [int]$LocalPort = 18867,
    [int]$RemotePort = 8765,
    [string]$RemoteRepository = '/staff/tangjiarui/OPC/spaceqa-agent',
    [switch]$StartRemoteService
)

$ErrorActionPreference = 'Stop'
$Url = "http://127.0.0.1:$LocalPort/evaluation.html"

# The packaged launcher can ask this script to start the managed service first.
# No dataset, model or key is copied to the client computer.
if ($StartRemoteService) {
    if ($RemoteRepository -notmatch '^/[A-Za-z0-9_./-]+$') {
        throw 'RemoteRepository must be an absolute server path without shell metacharacters.'
    }
    Write-Host "Checking SpaceQA service on $Server ..."
    & ssh -o BatchMode=yes -o ConnectTimeout=15 $Server "bash $RemoteRepository/deploy/web_service/start_server.sh start"
    if ($LASTEXITCODE -ne 0) {
        throw "The SpaceQA service could not be started on $Server. Check SSH access and RemoteRepository."
    }
}
try {
    $existing = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing
    if ($existing.StatusCode -eq 200) {
        Write-Host "SpaceQA evaluation console is already available: $Url"
        Start-Process $Url
        return
    }
} catch { }

Write-Host "Forwarding 127.0.0.1:$LocalPort -> $Server`:127.0.0.1:$RemotePort"
Write-Host "Open $Url after the tunnel is established. Keep this window open."
& ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    -N -T -L "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}" $Server
if ($LASTEXITCODE -ne 0) {
    throw "SSH forwarding ended. Check SSH access to $Server and whether local port $LocalPort is occupied."
}
