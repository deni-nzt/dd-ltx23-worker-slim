# Stage-1 scale patch (2026-08-12 mouth-resolution research) — the SAME bytes boot.sh
# applies at runtime, applied at BUILD time here. String-exact against pin 4f890573;
# a missed anchor keeps stock behavior and fails the BUILD loudly (unlike the boot-time
# variant, which degrades — at build time we want to know).
import sys

p = sys.argv[1]
s = open(p).read()
if "LTX_STAGE1_SCALE" in s:
    print("stage1-scale patch: already applied")
    sys.exit(0)

old_shape = """        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )"""
new_shape = """        _s1s = float(__import__("os").environ.get("LTX_STAGE1_SCALE", "0.5"))
        _snap32 = lambda v: max(32, int(round(v / 32)) * 32)
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=_snap32(width * _s1s),
            height=_snap32(height * _s1s),
            fps=frame_rate,
        )"""
old_up = "        upscaled_video_latent = self.upsampler(video_state.latent[:1])"
new_up = """        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        _th, _tw = height // 32, width // 32
        if upscaled_video_latent.shape[-2:] != (_th, _tw):
            _b, _c, _f, _h, _w = upscaled_video_latent.shape
            _lat = upscaled_video_latent.permute(0, 2, 1, 3, 4).reshape(_b * _f, _c, _h, _w)
            _lat = torch.nn.functional.interpolate(_lat, size=(_th, _tw), mode="bilinear", align_corners=False)
            upscaled_video_latent = _lat.reshape(_b, _f, _c, _th, _tw).permute(0, 2, 1, 3, 4)"""

if old_shape not in s or old_up not in s:
    print("stage1-scale patch: ANCHORS NOT FOUND in " + p)
    sys.exit(1)

open(p, "w").write(s.replace(old_shape, new_shape).replace(old_up, new_up))
print("stage1-scale patch: APPLIED to " + p)
