"""Smoke tests for scene-change frame extraction."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import extract_scene_change, extract, drop_flash_shots  # noqa: E402


class TestDropFlashShots(unittest.TestCase):

    def test_keeps_well_spaced_shots(self):
        self.assertEqual(drop_flash_shots([0.0, 2.0, 4.0], min_shot_seconds=0.5), [0, 1, 2])

    def test_drops_opening_flash_keeps_later_frame(self):
        # Frame 0 is forced in by eq(n,0); a real cut lands 0.1s later.
        self.assertEqual(drop_flash_shots([0.0, 0.1, 7.2], min_shot_seconds=0.5), [1, 2])

    def test_collapses_run_of_flashes_to_last(self):
        self.assertEqual(
            drop_flash_shots([0.0, 3.0, 3.1, 3.2, 3.3, 9.0], min_shot_seconds=0.5),
            [0, 4, 5],
        )

    def test_last_shot_always_kept(self):
        self.assertEqual(drop_flash_shots([0.0, 5.0, 5.2], min_shot_seconds=0.5), [0, 2])

    def test_empty_and_single(self):
        self.assertEqual(drop_flash_shots([], min_shot_seconds=0.5), [])
        self.assertEqual(drop_flash_shots([1.0], min_shot_seconds=0.5), [0])

    def test_zero_gap_disables_filter(self):
        self.assertEqual(drop_flash_shots([0.0, 0.1], min_shot_seconds=0.0), [0, 1])


def _make_test_video(out: Path, seconds: int = 6) -> Path:
    """Generate a synthetic test video with 3 distinct scenes (color changes)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=red:size=320x240:duration=2",
        "-f", "lavfi", "-i", f"color=c=green:size=320x240:duration=2",
        "-f", "lavfi", "-i", f"color=c=blue:size=320x240:duration=2",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


class TestSceneChange(unittest.TestCase):

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-test-"))
        self.video = _make_test_video(self.tmp / "input.mp4", seconds=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detects_scene_boundaries(self):
        out_dir = self.tmp / "frames"
        frames = extract_scene_change(
            str(self.video), out_dir,
            scene_threshold=0.3, resolution=128, max_frames=10,
        )
        self.assertGreaterEqual(len(frames), 2, "expected >=2 scene frames")
        self.assertLessEqual(len(frames), 10, "respect max_frames cap")
        for f in frames:
            self.assertTrue(Path(f["path"]).exists())
            self.assertIn("timestamp_seconds", f)

    def test_opening_flash_is_not_a_separate_shot(self):
        # 0.1s red flash, then 3 real 2s scenes — mirrors reels with a cover flash.
        flashy = self.tmp / "flashy.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=30:duration=0.1",
            # Distinct luma per scene: ffmpeg's scene score is luma-based, so
            # hue-only changes (e.g. red->green) would not register as cuts.
            "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=2",
            "-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=30:duration=2",
            "-f", "lavfi", "-i", "color=c=white:size=320x240:rate=30:duration=2",
            "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(flashy),
        ], check=True, capture_output=True)

        out_dir = self.tmp / "flashy_frames"
        frames = extract_scene_change(
            str(flashy), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=10, uniform_fallback_min=1,
        )
        self.assertEqual(len(frames), 3, [f["timestamp_seconds"] for f in frames])
        self.assertGreater(frames[0]["timestamp_seconds"], 0.0)
        # Files are renumbered contiguously after dropping the flash.
        names = [Path(f["path"]).name for f in frames]
        self.assertEqual(names, ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"])
        self.assertEqual(len(list(out_dir.glob("frame_*.jpg"))), 3)

    def test_falls_back_to_uniform_when_no_scenes(self):
        static = self.tmp / "static.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=320x240:duration=4",
            "-pix_fmt", "yuv420p",
            str(static),
        ], check=True, capture_output=True)

        out_dir = self.tmp / "static_frames"
        frames = extract_scene_change(
            str(static), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=10, uniform_fallback_min=5,
        )
        self.assertGreaterEqual(len(frames), 5, "fallback should produce >=5 frames")


if __name__ == "__main__":
    unittest.main()
