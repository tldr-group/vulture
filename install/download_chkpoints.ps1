# download checkpoints from huggingface with curl
# 1) get json of repo 2) filter to .pth 3) download 
$Username = "rmdocherty"
$ModelId = "vulture"
$TargetDir = "trained_models"
# this is a token with read-only access to the repo
# $HF_TOKEN = "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"

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null


# $headers = @{ Authorization = "Bearer $HF_TOKEN" }
$response = (Invoke-RestMethod "https://huggingface.co/api/models/$Username/$ModelId")
$files = $response.siblings.rfilename | Where-Object { $_ -match '\.pth$' }

foreach ($file in $files) {
    $url = "https://huggingface.co/$Username/$ModelId/resolve/main/$file"
    $outPath = Join-Path $TargetDir $file
    $outDir = Split-Path $outPath
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "Downloading $file ..."
    Invoke-WebRequest -Uri $url -OutFile $outPath
}