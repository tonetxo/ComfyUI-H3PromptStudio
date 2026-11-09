import os
import shlex
import subprocess
import time
import requests

try:
    from aiohttp import web
except ImportError:
    web = None

try:
    from .nodes import DEFAULT_LLAMACPP_URL, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, PRESETS_DATA, load_local_settings
except (ImportError, ValueError):
    from nodes import DEFAULT_LLAMACPP_URL, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, PRESETS_DATA, load_local_settings

WEB_DIRECTORY = "./web"


def _rescan_and_restart() -> dict:
    """Run the optional llamacpp_rescan_command setting, then report what the server lists."""
    settings = load_local_settings()
    command = (settings.get("llamacpp_rescan_command") or "").strip()
    if command:
        try:
            subprocess.run([os.path.expanduser(arg) for arg in shlex.split(command)], check=True, timeout=180)
        except subprocess.CalledProcessError as e:
            return {"ok": False, "message": f"Rescan failed: {e}"}
        except Exception as e:
            return {"ok": False, "message": f"Rescan error: {e}"}

    # A command may have restarted the server, so give it time to come back up.
    base_url = (settings.get("llamacpp_url") or DEFAULT_LLAMACPP_URL).rstrip("/")
    for _ in range(40 if command else 1):
        try:
            r = requests.get(base_url + "/v1/models", timeout=2)
            if r.ok:
                data = r.json()
                ids = [m.get("id", "").strip() for m in data.get("data", []) if m.get("id", "").strip()]
                return {"ok": True, "message": f"llama.cpp OK — {len(ids)} model(s): {', '.join(ids)}"}
        except Exception:
            pass
        if command:
            time.sleep(0.5)
    return {"ok": False, "message": f"llama.cpp did not respond at {base_url}"}


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

    @PromptServer.instance.routes.get("/h3promptstudio/presets")
    async def h3_presets(request):
        # Preset texts, used by the web UI as placeholders for empty custom_* fields.
        return web.json_response(PRESETS_DATA)


register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
