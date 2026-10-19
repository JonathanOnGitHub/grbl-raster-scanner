#!/usr/bin/env python3
"""One-shot test: run droplet_v2.analyze on a single image."""
import sys
import cv2
import droplet_v2 as dv2
import droplet_cv as dc

p = sys.argv[1] if len(sys.argv) > 1 else "snapshots/live_fresh.png"
bgr = cv2.imread(p, cv2.IMREAD_COLOR)
if bgr is None:
    sys.exit(f"could not read {p}")
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
gray_s = cv2.GaussianBlur(gray, (3, 3), 0)
res = dv2.analyze(bgr)
out = p.replace(".png", "_v2_dbg.png")
if res.debug is not None:
    cv2.imwrite(out, res.debug)
print(f"== {p} ==")
print(f"wrote {out}")
print(f"tube_ok         = {res.tube.ok}")
print(f"tube_W_px       = {res.tube.width_px:.2f}")
print(f"tube_reason     = {res.tube.reason!r}")
print(f"tip             = ({res.tube.tip[0]:.1f}, {res.tube.tip[1]:.1f})")
print(f"tip_end         = {res.tube.tip_end:+d}")
print(f"body_dir        = ({res.tube.body_dir[0]:+.3f}, {res.tube.body_dir[1]:+.3f})")
print(f"droplet_ok      = {res.droplet.ok}")
print(f"droplet_D_W     = {res.droplet.diameter_in_tube_widths:.3f}")
print(f"droplet_h_W     = {res.droplet.vertical_extent_in_tube_widths:+.3f}")
print(f"droplet_area_px = {res.droplet.area_px:.1f}")
print(f"droplet_reason  = {res.droplet.reason!r}")
# Also dump per-orientation candidates so we can see both orientations
axis = dc._find_tube_axis_bright_col(gray_s)
if axis is not None:
    p0, p1, _ = axis
    print(f"\naxis p0={p0} p1={p1}")
    for ori in ("p0_tip", "p1_tip"):
        cand = dv2._try_orientation(gray, p0, p1, ori, ux=(p1[0]-p0[0])/((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)**0.5, uy=(p1[1]-p0[1])/((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)**0.5, L=((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)**0.5)
        d = cand["droplet"]
        print(f"  {ori}: tip={cand['tip']} W={cand['w_tube']:.1f} term={cand['terminus_score']:.2f} drop_ok={d.ok} area={d.area_px:.0f} D_W={d.diameter_in_tube_widths:.2f} h_W={d.vertical_extent_in_tube_widths:+.2f}")
