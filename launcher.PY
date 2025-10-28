# launcher.py
import os, sys, time, socket, webbrowser, threading
from pathlib import Path

# >>> ajuste aqui se seu arquivo tiver outro nome
APP_FILENAME = "app_streamlit.py"

OPEN_BROWSER = True   # coloque False se NÃO quiser abrir o navegador automaticamente
HOST = "127.0.0.1"
DEFAULT_PORT = 8501

def resource_path(rel: str) -> str:
    # resolve caminho quando empacotado (PyInstaller) ou em desenvolvimento
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent
    return str((base / rel).resolve())

def pick_free_port(start=DEFAULT_PORT, end=DEFAULT_PORT+20) -> int:
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, p))
                return p
            except OSError:
                continue
    return DEFAULT_PORT

def wait_port(port: int, timeout: int = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((HOST, port))
                return True
            except OSError:
                time.sleep(0.3)
    return False

def main():
    app_path = resource_path(APP_FILENAME)
    if not Path(app_path).exists():
        candidate = Path.cwd() / APP_FILENAME
        if candidate.exists():
            app_path = str(candidate.resolve())

    port = int(os.environ.get("PORT", pick_free_port()))

    # >>> IMPORTANTE: desliga o modo dev para poder setar a porta
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENTMODE"] = "false"

    args = [
        "streamlit", "run", app_path,
        "--global.developmentMode", "false",      # <— aqui
        "--server.headless", "true",
        "--server.address", HOST,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]

    if OPEN_BROWSER:
        def _open():
            if wait_port(port, 30):
                webbrowser.open_new_tab(f"http://{HOST}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    from streamlit.web.cli import main as stcli
    sys.argv = args
    sys.exit(stcli())


if __name__ == "__main__":
    main()
