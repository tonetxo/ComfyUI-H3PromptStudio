import os
import sys
import subprocess
import time
import requests

try:
    from aiohttp import web
except ImportError:
    web = None

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, ValueError):
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

RESCAN_SCRIPT = "/home/tonetxo/.config/h3ps/rescan_llamacpp_models.py"


def _rescan_and_restart() -> dict:
    """Regenerate the llamacpp router preset and restart the service."""
    try:
        if os.path.exists(RESCAN_SCRIPT):
            subprocess.run([sys.executable, RESCAN_SCRIPT], check=True, timeout=60)
        subprocess.run(
            ["systemctl", "--user", "restart", "h3ps-llamacpp.service"],
            check=True,
            timeout=180,
        )
    except subprocess.CalledProcessError as e:
        return {"ok": False, "message": f"Service restart failed: {e}"}
    except Exception as e:
        return {"ok": False, "message": f"Rescan error: {e}"}

    url = "http://127.0.0.1:8080/v1/models"
    for _ in range(40):
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                data = r.json()
                ids = [m.get("id", "").strip() for m in data.get("data", []) if m.get("id", "").strip()]
                return {"ok": True, "message": f"llama.cpp OK — {len(ids)} model(s): {', '.join(ids)}"}
        except Exception:
            pass
        time.sleep(0.5)
    return {"ok": False, "message": "llama.cpp router restart timed out"}


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        return
    if PromptServer.instance is None:
        return

    @PromptServer.instance.routes.post("/h3promptstudio/rescan_llamacpp")
    async def h3_rescan_llamacpp(request):
        if web is not None:
            return web.json_response(_rescan_and_restart())
        return web.json_response({"ok": False, "message": "aiohttp unavailable"})


register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
