import os
import subprocess
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from classpulse.backend import create_app

app = create_app()


def get_start_url(host: str = "localhost", port: int = 5000, path: str = "/login", use_https: bool = False) -> str:
    resolved_host = host if host not in {"0.0.0.0", "::", "127.0.0.1"} else "localhost"
    scheme = "https" if use_https else "http"
    return f"{scheme}://{resolved_host}:{port}{path}"


def open_browser_to_login(url: str) -> None:
    candidates = [
        os.getenv("CHROME_BIN"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        rf"{os.getenv('LOCALAPPDATA', '')}\Google\Chrome\Application\chrome.exe",
    ]

    chrome_path = next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)
    if chrome_path:
        try:
            subprocess.Popen(
                [
                    chrome_path,
                    "--new-window",
                    "--ignore-certificate-errors",
                    "--allow-insecure-localhost",
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
            return
        except Exception:
            pass

    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass


def trust_local_certificate_for_windows(cert_path: str | Path) -> None:
    if os.name != "nt":
        return
    cert_file = Path(cert_path)
    if not cert_file.exists():
        return

    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"Import-Certificate -FilePath '{str(cert_file)}' -CertStoreLocation 'Cert:\\CurrentUser\\Root' | Out-Null",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def ensure_https_certificate(cert_path: str | None, key_path: str | None) -> tuple[str, str] | None:
    project_root = Path(__file__).resolve().parent
    cert_file = Path(cert_path) if cert_path else project_root / "certs" / "cert.pem"
    key_file = Path(key_path) if key_path else project_root / "certs" / "key.pem"

    cert_file.parent.mkdir(parents=True, exist_ok=True)

    if cert_file.exists() and key_file.exists():
        trust_local_certificate_for_windows(cert_file)
        return str(cert_file), str(key_file)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.DNSName("127.0.0.1")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    trust_local_certificate_for_windows(cert_file)
    return str(cert_file), str(key_file)


if __name__ == "__main__":
    cert_path = os.getenv("SSL_CERT_PATH")
    key_path = os.getenv("SSL_KEY_PATH")
    ssl_context = None
    enable_https = os.getenv("ENABLE_HTTPS", "0") == "1"
    auto_open_browser = os.getenv("AUTO_OPEN_BROWSER", "1") == "1"
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "5000"))
    start_url = get_start_url(host=host, port=port, use_https=enable_https)

    if enable_https:
        print("HTTPS mode is enabled. For a local laptop webcam, keep ENABLE_HTTPS=0 unless you have already trusted the generated localhost certificate.")
        ssl_pair = ensure_https_certificate(cert_path, key_path)
        if ssl_pair:
            ssl_context = ssl_pair
        else:
            print("HTTPS enabled but certificate generation failed. Running without SSL.")
    else:
        print("Local camera mode is active. Open http://localhost:5000/login to allow the laptop webcam in the browser.")

    print(f"ClassPulse is running locally at: {start_url}")
    if auto_open_browser:
        threading.Timer(1.5, open_browser_to_login, args=(start_url,)).start()

    app.run(
        host=host,
        port=port,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        ssl_context=ssl_context,
    )
