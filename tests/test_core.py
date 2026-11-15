import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(module_name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    added_stubs = []
    for pkg in ("torch", "numpy"):
        if pkg not in sys.modules and importlib.util.find_spec(pkg) is None:
            stub = types.ModuleType(pkg)
            if pkg == "torch":
                stub.Tensor = object
            sys.modules[pkg] = stub
            added_stubs.append(pkg)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for pkg in added_stubs:
            sys.modules.pop(pkg, None)


h3_gui = _load_module("h3_gui_under_test", "h3_gui.py")
root_nodes = _load_module("root_nodes_under_test", "nodes.py")
studio_nodes = _load_module(
    "studio_nodes_under_test",
    "custom_nodes/ComfyUI-H3PromptStudio/nodes.py",
)
# The Qwen rewriter node is local-only (gitignored), so it is tested only when present.
_QWEN_PATH = "custom_nodes/ComfyUI-Qwen-Prompt-Rewriter/nodes.py"
qwen_nodes = _load_module("qwen_nodes_under_test", _QWEN_PATH) if (REPO_ROOT / _QWEN_PATH).exists() else None


class SanitizeUrlTests(unittest.TestCase):
    def test_sanitize_url_across_all_modules(self):
        modules = [
            (h3_gui, "http://127.0.0.1:11434"),
            (root_nodes, "http://127.0.0.1:11434"),
            (studio_nodes, "http://127.0.0.1:11434"),
        ]
        if qwen_nodes is not None:
            modules.append((qwen_nodes, "http://127.0.0.1:11434"))
        for mod, default in modules:
            with self.subTest(module=mod.__name__):
                self.assertEqual(mod.sanitize_url("https://llm.local:8443///", default), "https://llm.local:8443")
                self.assertEqual(mod.sanitize_url("  http://127.0.0.1:8080/ ", default), "http://127.0.0.1:8080")
                self.assertEqual(mod.sanitize_url("", default), default)
                self.assertEqual(mod.sanitize_url("127.0.0.1:8080", default), "http://127.0.0.1:8080")
                for bad in ("file:///etc/passwd", "ftp://example.com/model", "http://", "javascript:alert(1)"):
                    with self.assertRaises(ValueError):
                        mod.sanitize_url(bad, default)


class JsonExtractionTests(unittest.TestCase):
    def test_extract_json_and_repair_across_modules(self):
        for mod in (h3_gui, root_nodes, studio_nodes):
            with self.subTest(module=mod.__name__):
                parsed = mod.extract_json('<think>plan</think>\n```json\n{"shot": 1, "ok": true,}\n```')
                self.assertEqual(parsed, {"shot": 1, "ok": True})

                truncated = '{"shots": [{"id": "S1"}, {"id": "S2"'
                repaired = mod._repair_json(truncated)
                self.assertIsInstance(repaired, dict)
                self.assertIn("shots", repaired)
                self.assertEqual(len(repaired["shots"]), 2)

                fallback = mod.extract_json("plain text without json", fallback_key="fallback")
                self.assertEqual(fallback["fallback"], "plain text without json")
                self.assertEqual(fallback["shots"], [])


class EnhancerTests(unittest.TestCase):
    def test_clean_and_validate_enhancer_output(self):
        raw = (
            "<thinking>rough draft</thinking>\n"
            "subject_definitions:\n- draft\n"
            "Now construct the full response\n"
            "integrated_multimodal_description:\nA cinematic rain-slicked alley at night."
        )
        cleaned = h3_gui.clean_enhancer_output(raw)
        self.assertTrue(cleaned.startswith("integrated_multimodal_description:"))
        self.assertNotIn("draft", cleaned)
        self.assertTrue(h3_gui.is_enhancer_output_valid("T2VA", cleaned))
        self.assertFalse(h3_gui.is_enhancer_output_valid("R2V", cleaned))
        self.assertEqual(h3_gui.clean_enhancer_output("no header reasoning only"), "")


class EvolveTests(unittest.TestCase):
    def test_generate_evolved_deterministic(self):
        out1 = h3_gui.generate_evolved("A dark slow river moves smoothly.", "synonyms", 50, 2, 42)
        out2 = h3_gui.generate_evolved("A dark slow river moves smoothly.", "synonyms", 50, 2, 42)
        out3 = h3_gui.generate_evolved("A dark slow river moves smoothly.", "synonyms", 50, 2, 43)
        self.assertEqual(out1, out2)
        self.assertNotEqual(out1, out3)

    def test_evolve_synonyms_probability_not_squared(self):
        keys = [k for k in h3_gui.EVOLVE_SYNONYMS.keys()][:5]
        self.assertGreaterEqual(len(keys), 3)
        sample_prompt = " ".join([k.capitalize() for k in keys])

        # strength=100 must replace 100% of synonym tokens and preserve capitalization.
        full = h3_gui.evolve_synonyms(sample_prompt, 100, rng=random.Random(7))
        for word, orig_key in zip(full.split(), keys):
            self.assertIn(word.lower(), h3_gui.EVOLVE_SYNONYMS[orig_key])
            self.assertTrue(word[0].isupper())

        # strength=50 over 400 tokens should yield ~50% replaced (not ~25%).
        long_prompt = " ".join(keys * 80)
        half = h3_gui.evolve_synonyms(long_prompt, 50, rng=random.Random(123))
        replaced = sum(1 for orig, new in zip(long_prompt.split(), half.split()) if orig != new)
        ratio = replaced / len(long_prompt.split())
        self.assertGreater(ratio, 0.40)
        self.assertLess(ratio, 0.60)


class OnerHelpersTests(unittest.TestCase):
    def test_align_parse_and_warnings(self):
        self.assertEqual(h3_gui.align_h3_frames(200), 209)  # 5 + 17 * 12 = 209
        self.assertEqual(h3_gui.align_h3_frames(9999), h3_gui.ONER_MAX_FRAMES)

        self.assertEqual(h3_gui.parse_oner_frames("107, 141 175"), [107, 141, 175])
        with self.assertRaises(ValueError):
            h3_gui.parse_oner_frames("107, abc")
        with self.assertRaises(ValueError):
            h3_gui.parse_oner_frames("0, 107")

        raw = (
            "<think>plan</think>\n"
            "=== SEGMENT 1 ===\n"
            "Subject stays with <Picture 1> ||| [Style] dark\n"
            "=== SEGMENT 2 ===\n"
            "integrated_multimodal_description:\n"
            "overall_soundscape:\n"
            "non_diegetic_music:\n"
            "[Shot 02] camera cuts to <Picture 5>\n"
        )
        segs = h3_gui.parse_oner_segments(raw)
        self.assertEqual(len(segs), 2)
        self.assertNotIn("|||", segs[0])

        warnings = h3_gui.oner_warnings(segs, expected=3, ref_count=2)
        joined = " | ".join(warnings)
        self.assertIn("se esperaban 3 segmentos", joined)
        self.assertIn("al segmento 1 le falta", joined)
        self.assertIn("el segmento 2 tiene más de un plano", joined)
        self.assertIn("cita imágenes que no existen (5)", joined)

    def test_parse_oner_txt(self):
        txt = (
            "=== MAIN PROMPT ===\nMain text\n\n"
            "=== SECTION PROMPTS ===\nSec 2\n|||\nSec 3\n\n"
            "=== SECTION DURATIONS ===\n107,107,107\n"
        )
        main, secs, durs = h3_gui.parse_oner_txt(txt)
        self.assertEqual(main, "Main text")
        self.assertEqual(secs, "Sec 2\n|||\nSec 3")
        self.assertEqual(durs, "107,107,107")


class MotionAndReferenceHelpersTests(unittest.TestCase):
    def test_motion_plan_to_text_handles_strings_lists_dicts(self):
        plan = {
            "primary_actions": "Walks across room",
            "camera_motion": {"type": "dolly in", "speed": "slow"},
            "timing_beats": ["0s enter", "4s pause"],
            "motion_constraints": "Keep face steady",
            "audio_events": ["footsteps"],
        }
        text = h3_gui._motion_plan_to_text(plan)
        self.assertIn("Primary actions: Walks across room", text)
        self.assertIn("Camera: dolly in slow", text)
        self.assertIn("Timing: 0s enter 4s pause", text)
        self.assertIn("Constraints: Keep face steady", text)
        self.assertIn("Audio events: footsteps", text)
        self.assertEqual(h3_gui._motion_plan_to_text(None), "")

    def test_parse_seconds_all_modules(self):
        for mod in (h3_gui, root_nodes, studio_nodes):
            self.assertEqual(mod.parse_seconds("10s", 5), 10)
            self.assertEqual(mod.parse_seconds("8.6s", 5), 9)
            self.assertEqual(mod.parse_seconds("invalid", 12), 12)

    def test_reference_labels_and_tag_enforcement(self):
        labels_txt = "hero.png | CHARACTER | Mara\ncity.png | ENVIRONMENT\nvillain.png"
        paths = ["/tmp/hero.png", "/tmp/city.png", "/tmp/villain.png", "/tmp/neon_alley.jpg"]
        lib = h3_gui.parse_reference_labels(labels_txt, paths)
        self.assertEqual([r["tag"] for r in lib], ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"])
        self.assertEqual((lib[0]["role"], lib[0]["label"]), ("CHARACTER", "Mara"))
        self.assertEqual((lib[1]["role"], lib[1]["label"]), ("ENVIRONMENT", "city.png"))
        self.assertEqual((lib[2]["role"], lib[2]["label"]), ("GENERAL", "villain.png"))
        self.assertEqual((lib[3]["role"], lib[3]["label"]), ("GENERAL", "neon alley"))

        tags = h3_gui._director_shot_tags({"reference_tags": ["<Picture 2>", "99", "picture 1"]}, lib)
        self.assertEqual(tags, ["<Picture 2>", "<Picture 1>"])
        fallback_tags = h3_gui._director_shot_tags({"reference_tags": ["99"]}, lib)
        self.assertEqual(fallback_tags, ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"])

        prompt = h3_gui._ensure_reference_tags("Mara runs down the street.", ["<Picture 1>", "<Picture 2>"], lib)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)

        for mod in (root_nodes, studio_nodes):
            p = mod.ensure_reference_tags("Action shot", ["<Picture 1>", "<Picture 3>"])
            self.assertIn("<Picture 1>", p)
            self.assertIn("<Picture 3>", p)
            self.assertEqual(mod.ensure_reference_tags("Already has <Picture 1>.", ["<Picture 1>"]), "Already has <Picture 1>.")

    def test_unload_skips_llamacpp_and_qwen_settings_fallback(self):
        with mock.patch.object(h3_gui.requests, "get") as mock_get, mock.patch.object(h3_gui.requests, "post") as mock_post:
            h3_gui.ollama_unload_all("http://127.0.0.1:8080", backend="llamacpp")
            h3_gui.ollama_unload_model("http://127.0.0.1:8080", "qwen", backend="llamacpp")
            mock_get.assert_not_called()
            mock_post.assert_not_called()

        if qwen_nodes is None:
            return
        with mock.patch.object(qwen_nodes, "_load_local_settings", return_value={"llamacpp_url": "http://127.0.0.1:9090/", "gguf_models_dir": "~/custom-gguf"}):
            self.assertEqual(qwen_nodes._resolve_backend_url("http://127.0.0.1:11434", "llamacpp"), "http://127.0.0.1:9090")
            self.assertEqual(qwen_nodes._get_gguf_models_dir(), Path("~/custom-gguf").expanduser())


if __name__ == "__main__":
    unittest.main()
