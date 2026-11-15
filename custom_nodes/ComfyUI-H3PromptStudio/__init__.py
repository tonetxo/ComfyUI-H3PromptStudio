import asyncio
import os
import shlex
import subprocess
import threading
import time
import requests

try:
    from aiohttp import web
except ImportError:
    web = None

try:
    from .nodes import (
        DEFAULT_LLAMACPP_URL,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        PRESETS_DATA,
        invalidate_model_cache,
        load_local_settings,
        sanitize_url,
    )
except (ImportError, ValueError):
    from nodes import (
        DEFAULT_LLAMACPP_URL,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        PRESETS_DATA,
        invalidate_model_cache,
        load_local_settings,
        sanitize_url,
    )

WEB_DIRECTORY = "./web"
_RESCAN_LOCK = threading.Lock()


def _rescan_and_restart() -> dict:
    """Run the optional llamacpp_rescan_command setting, then report what the server lists."""
    if not _RESCAN_LOCK.acquire(blocking=False):
        return {"ok": False, "message": "Rescan already in progress"}
    try:
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
        try:
            base_url = sanitize_url(settings.get("llamacpp_url"), default=DEFAULT_LLAMACPP_URL)
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        for _ in range(40 if command else 1):
            try:
                r = requests.get(base_url + "/v1/models", timeout=2)
                if r.ok:
                    data = r.json()
                    ids = [m.get("id", "").strip() for m in data.get("data", []) if m.get("id", "").strip()]
                    invalidate_model_cache()
                    return {
                        "ok": True,
                        "models": ids,
                        "message": f"llama.cpp OK — {len(ids)} model(s): {', '.join(ids)}",
                    }
            except Exception:
                pass
            if command:
                time.sleep(0.5)
        return {"ok": False, "message": f"llama.cpp did not respond at {base_url}"}
    finally:
        _RESCAN_LOCK.release()


def register_routes():
    if web is None:
        return
    try:
        from server import PromptServer
    except ImportError:
        return
    if PromptServer.instance is None:
        return

    @PromptServer.instance.routes.post("/h3promptstudio/rescan_llamacpp")
    async def h3_rescan_llamacpp(request):
        data = await asyncio.to_thread(_rescan_and_restart)
        return web.json_response(data)

    @PromptServer.instance.routes.get("/h3promptstudio/presets")
    async def h3_presets(request):
        # Preset texts, used by the web UI as placeholders for empty custom_* fields.
        return web.json_response(PRESETS_DATA)


register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
