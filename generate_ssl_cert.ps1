$certDir = Join-Path $PSScriptRoot "certs"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null

$certPath = Join-Path $certDir "cert.pem"
$keyPath = Join-Path $certDir "key.pem"

if (-not (Test-Path $certPath) -or -not (Test-Path $keyPath)) {
    $pythonExecutable = if (Test-Path (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")) {
        Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        "python"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        "py"
    } else {
        $null
    }

    if (-not $pythonExecutable) {
        Write-Error "Python is required to generate a self-signed HTTPS certificate. Install Python and then rerun this script."
        exit 1
    }

    $pythonCode = @"
from pathlib import Path
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

cert_dir = Path(r'$certDir')
cert_path = cert_dir / 'cert.pem'
key_path = cert_dir / 'key.pem'

if cert_path.exists() and key_path.exists():
    raise SystemExit(0)

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, 'localhost'),
])
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
    .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False)
    .sign(key, hashes.SHA256())
)

cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
key_path.write_bytes(key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
))
print(f'Generated {cert_path}')
print(f'Generated {key_path}')
"@

    & $pythonExecutable -c $pythonCode
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Unable to generate the self-signed certificate. Ensure cryptography is installed in the selected Python environment."
        exit 1
    }
}

Write-Host "Certificate ready at $certPath"
Write-Host "Private key ready at $keyPath"
