param(
    [Parameter(Mandatory = $true)]
    [string]$PineconeApiKey,
    [string]$SecretName = "awspineconeapikey1",
    [string]$Region = "us-west-1"
)

$existingSecret = aws secretsmanager describe-secret `
    --region $Region `
    --secret-id $SecretName 2>$null

if ($LASTEXITCODE -eq 0 -and $existingSecret) {
    aws secretsmanager put-secret-value `
        --region $Region `
        --secret-id $SecretName `
        --secret-string $PineconeApiKey
} else {
    aws secretsmanager create-secret `
        --region $Region `
        --name $SecretName `
        --secret-string $PineconeApiKey
}

