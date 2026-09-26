"""Smoke tests for scene-change frame extraction."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import extract_scene_change, extract, drop_flash_shots, gap_fill_times  # noqa: E402


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


class TestGapFillTimes(unittest.TestCase):

    def test_no_fill_when_all_shots_short(self):
        self.assertEqual(gap_fill_times([0.0, 5.0, 10.0], range_end=15.0, max_gap=8.0, budget=10), [])

    def test_long_shot_split_evenly(self):
        # 31s shot, max gap 8 -> 4 even pieces -> 3 extra frames.
        times = gap_fill_times([0.0, 31.0], range_end=35.0, max_gap=8.0, budget=10)
        self.assertEqual([round(t, 2) for t in times], [7.75, 15.5, 23.25])

    def test_tail_after_last_cut_is_filled(self):
        times = gap_fill_times([0.0], range_end=20.0, max_gap=8.0, budget=10)
        self.assertEqual([round(t, 2) for t in times], [6.67, 13.33])

    def test_budget_caps_extras_and_widens_spacing(self):
        times = gap_fill_times([0.0, 31.0], range_end=35.0, max_gap=8.0, budget=1)
        self.assertEqual([round(t, 2) for t in times], [15.5])

    def test_zero_budget_or_disabled(self):
        self.assertEqual(gap_fill_times([0.0, 31.0], range_end=35.0, max_gap=8.0, budget=0), [])
        self.assertEqual(gap_fill_times([0.0, 31.0], range_end=35.0, max_gap=0.0, budget=10), [])

    def test_empty_input(self):
        self.assertEqual(gap_fill_times([], range_end=35.0, max_gap=8.0, budget=10), [])


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

    def test_long_shot_gets_gap_fill_frames(self):
        # black 2s | gray 20s (one long shot) | white 2s
        longshot = self.tmp / "longshot.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=2",
            "-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=30:duration=20",
            "-f", "lavfi", "-i", "color=c=white:size=320x240:rate=30:duration=2",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(longshot),
        ], check=True, capture_output=True)

        out_dir = self.tmp / "longshot_frames"
        frames = extract_scene_change(
            str(longshot), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=10, uniform_fallback_min=1, max_shot_gap=8.0,
        )
        sources = [f["source"] for f in frames]
        self.assertEqual(sources.count("scene-change"), 3, sources)
        self.assertEqual(sources.count("gap-fill"), 2, sources)
        times = [f["timestamp_seconds"] for f in frames]
        self.assertEqual(times, sorted(times))
        gap = [f["timestamp_seconds"] for f in frames if f["source"] == "gap-fill"]
        self.assertTrue(all(2.0 < t < 22.0 for t in gap), gap)
        names = [Path(f["path"]).name for f in frames]
        self.assertEqual(names, [f"frame_{i:04d}.jpg" for i in range(1, 6)])
        self.assertEqual(len(list(out_dir.glob("*.jpg"))), 5)

    def test_gap_fill_disabled(self):
        frames = extract_scene_change(
            str(self.video), self.tmp / "nofill",
            scene_threshold=0.3, resolution=128,
            max_frames=10, uniform_fallback_min=1, max_shot_gap=0.0,
        )
        self.assertTrue(all(f["source"] == "scene-change" for f in frames))

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
