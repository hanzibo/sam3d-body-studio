# SPDX-License-Identifier: GPL-3.0-only
"""Regression: different MP4 time bases must not stretch concatenated playback."""
import json,subprocess,tempfile,unittest
from pathlib import Path
from rgbd_comparison_playlist import make_playlist

class PlaylistTests(unittest.TestCase):
    def test_different_input_time_bases_keep_duration_and_every_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);videos=[]
            for i,scale in enumerate((1000,90000)):
                video=root/f'{i}.mp4';videos.append(video)
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=blue:s=64x64:r=10','-frames:v','10','-c:v','libx264','-threads','1','-video_track_timescale',str(scale),str(video)],check=True)
            make_playlist(root,videos)
            info=json.loads((root/'playlist.json').read_text())
            self.assertEqual(info['frames'],20)
            self.assertAlmostEqual(info['expected_duration_s'],2.,places=2)
            self.assertAlmostEqual(info['actual_duration_s'],2.,places=2)

if __name__=='__main__':unittest.main()
