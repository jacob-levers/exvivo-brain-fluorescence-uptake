#!/usr/bin/env python
"""Draw an ROI on the finished frame, save it, and render with --roi.

You draw on the exact image the movie will contain -- same crop, roll, zoom,
pan, window and colour -- so the polygon lands precisely where you put it.
Screen points are mapped back through the renderer's own geometry into the
volume, rasterised through every z-plane, and stored in full-cache pixel
coordinates so the same roi.json works for any crop.

    python scripts/roi_tool.py --cache-dir out6 --bin 2 --signal-channel 1 \
        --roll -3.3 --zoom 1.6 --pan-y 60 --width 900 --height 620 \
        --vmin 8 --vmax 72 --cmap green --out out6/roi.json

then open http://127.0.0.1:8780, draw, Save, and add  --roi out6/roi.json
to the render command.

Left-click adds a vertex; click near the first vertex (or "Close") to close a
polygon.  Several KEEP polygons are unioned; REMOVE polygons punch holes.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import socket
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd2_perfusion_3d as P

STATE = {}
LOCK = threading.Lock()


class Args:
    pass


def setup(a):
    args = Args()
    for k in ("cache_dir", "bin", "baseline_n", "bg_sigma", "bg_mode", "bg_pct",
              "signal_channel", "crop_pad_um"):
        setattr(args, k, getattr(a, k))
    args.outdir = a.cache_dir
    args.gpu = True
    args.force_bg = False

    arr, cmeta = P.load_cache(args)
    bg = P.compute_background(arr, cmeta, args)
    xp, ndi, _ = P.backend(True)
    nT, nC, nZ, Yb, Xb = arr.shape
    crop = None
    if a.crop:
        crop = P.sample_bbox(arr, bg, a.signal_channel, args, cmeta,
                             pad_um=a.crop_pad_um, frac=a.crop_frac)
        Yb, Xb = crop[1] - crop[0], crop[3] - crop[2]
    t = a.frame if a.frame >= 0 else nT - 1
    vol = P.build_volume(arr, bg, np.zeros((nT, 3)), t, [a.signal_channel],
                         xp, ndi, crop=crop)[0]
    rc = P.Raycaster((nZ, Yb, Xb), (cmeta["dz_um"], cmeta["dy_um"], cmeta["dx_um"]),
                     a.width, a.height, xp, ndi)
    lut_host = P.make_lut(a.cmap)
    STATE.update(a=a, cmeta=cmeta, xp=xp, ndi=ndi, vol=vol, rc=rc, crop=crop,
                 Yb=Yb, Xb=Xb, nZ=nZ, lut=xp.asarray(lut_host), lut_host=lut_host,
                 t=t)
    print(f"[roi] frame t={t}, crop {crop}, volume {Xb}x{Yb}x{nZ}")


def screen_to_voxel(px, py):
    """Invert the renderer's mapping for the top-down view (el=az=spin=0)."""
    a, rc = STATE["a"], STATE["rc"]
    W, H = a.width, a.height
    half_u = rc.radius / a.zoom
    half_v = half_u * (H / W)
    u = -half_u + px * (2 * half_u / (W - 1))
    v = half_v - py * (2 * half_v / (H - 1))
    r = math.radians(a.roll)
    cr, sr = math.cos(r), math.sin(r)
    U = u * cr - v * sr - a.pan_x
    V = u * sr + v * cr - a.pan_y
    cx = (U + rc.Lx / 2) / rc.dx
    cy = (V + rc.Ly / 2) / rc.dy
    return cx, cy


def render(mask=None):
    a, rc, xp = STATE["a"], STATE["rc"], STATE["xp"]
    vol = STATE["vol"] if mask is None else STATE["vol"] * mask[None, :, :]
    img = rc.render([vol], [STATE["lut"]], [a.vmin], [a.vmax], [a.opacity],
                    0.0, 0.0, n_samples=a.samples, mode=a.mode, zoom=a.zoom,
                    emit=[a.emit], roll=a.roll, pan=(a.pan_x, a.pan_y))
    img8 = (P.to_host(img) * 255).astype(np.uint8)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def rasterise(spec_polys_screen, feather_um):
    """Screen-space polygons -> float mask (Yb, Xb) in the cropped volume,
    plus the polygons in full-cache pixel coords for saving."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as sndi

    Yb, Xb = STATE["Yb"], STATE["Xb"]
    crop = STATE["crop"]
    y0, x0 = (crop[0], crop[2]) if crop is not None else (0, 0)
    im = Image.new("L", (Xb, Yb), 0)
    d = ImageDraw.Draw(im)
    keep_full, remove_full = [], []
    for kind, polys in (("keep", spec_polys_screen.get("keep", [])),
                        ("remove", spec_polys_screen.get("remove", []))):
        for poly in polys:
            vox = [screen_to_voxel(float(x), float(y)) for x, y in poly]
            if len(vox) >= 3:
                d.polygon(vox, fill=255 if kind == "keep" else 0)
                (keep_full if kind == "keep" else remove_full).append(
                    [[cx + x0, cy + y0] for cx, cy in vox])
    m = np.asarray(im, dtype=np.float32) / 255.0
    if feather_um > 0:
        m = sndi.gaussian_filter(m, feather_um / STATE["cmeta"]["dx_um"])
    return m, keep_full, remove_full


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>ROI tool</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif;display:flex;height:100vh}
#side{width:250px;padding:14px;background:#181818;border-right:1px solid #2c2c2c;overflow-y:auto}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:10px}
canvas{max-width:100%;max-height:94vh;cursor:crosshair;background:#000}
button{display:block;width:100%;margin:6px 0;padding:7px;background:#2a2a2a;color:#ddd;
       border:1px solid #3d3d3d;border-radius:4px;cursor:pointer}
button:hover{background:#333}
button.on{background:#2d5a2d;border-color:#4a8a4a}
button.rm{border-color:#7a3a3a}
button.rm.on{background:#5a2d2d}
label{display:block;margin:12px 0 3px;color:#9a9a9a;font-size:11px;text-transform:uppercase}
input[type=range]{width:100%}
#msg{margin-top:12px;color:#7fc98a;font:11px ui-monospace,monospace;white-space:pre-wrap;word-break:break-all}
.hint{color:#777;font-size:11px;line-height:1.5;margin-top:10px}
</style></head><body>
<div id="side">
<label>mode</label>
<button id="mKeep" class="on">KEEP (inside)</button>
<button id="mRem" class="rm">REMOVE (punch hole)</button>
<label>polygon</label>
<button id="close">Close polygon</button>
<button id="undo">Undo last point</button>
<button id="delpoly">Delete last polygon</button>
<button id="clear">Clear all</button>
<label>edge feather <span id="fv">15</span> um</label>
<input type="range" id="feather" min="0" max="60" step="1" value="15">
<label>output</label>
<button id="preview">Preview masked</button>
<button id="show">Show original</button>
<button id="save">Save ROI</button>
<div id="msg"></div>
<div class="hint">Click to add vertices. Click near the first vertex to close.
Draw one KEEP polygon around the brain; add REMOVE polygons for anything inside it you want gone.</div>
</div>
<div id="main"><canvas id="c" width="__W__" height="__H__"></canvas></div>
<script>
const W=__W__,H=__H__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const g=i=>document.getElementById(i);
let base=new Image(); base.src='/base.png?'+Date.now();
let shown=base;
let polys=[]; let cur=[]; let mode='keep';
base.onload=draw;
function pt(e){const r=cv.getBoundingClientRect();
  return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height];}
function draw(){
  ctx.clearRect(0,0,W,H); ctx.drawImage(shown,0,0,W,H);
  for(const p of polys) poly(p.pts,p.kind==='keep'?'#4dff88':'#ff5c5c',true);
  if(cur.length) poly(cur,mode==='keep'?'#4dff88':'#ff5c5c',false);
}
function poly(pts,col,closed){
  ctx.lineWidth=2; ctx.strokeStyle=col; ctx.fillStyle=col+'22';
  ctx.beginPath(); pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
  if(closed){ctx.closePath(); ctx.fill();} ctx.stroke();
  for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],3.5,0,6.283);ctx.fillStyle=col;ctx.fill();}
}
function closePoly(){ if(cur.length>=3){polys.push({kind:mode,pts:cur}); cur=[];} draw(); }
cv.addEventListener('click',e=>{
  const p=pt(e);
  if(cur.length>=3){const f=cur[0]; if(Math.hypot(p[0]-f[0],p[1]-f[1])<10){closePoly();return;}}
  cur.push(p); draw();});
g('close').onclick=closePoly;
g('undo').onclick=()=>{cur.pop(); draw();};
g('delpoly').onclick=()=>{polys.pop(); draw();};
g('clear').onclick=()=>{polys=[];cur=[]; shown=base; draw();};
g('mKeep').onclick=()=>{mode='keep';g('mKeep').classList.add('on');g('mRem').classList.remove('on');};
g('mRem').onclick=()=>{mode='remove';g('mRem').classList.add('on');g('mKeep').classList.remove('on');};
g('feather').oninput=()=>g('fv').textContent=g('feather').value;
function spec(){return {keep:polys.filter(p=>p.kind==='keep').map(p=>p.pts),
                        remove:polys.filter(p=>p.kind==='remove').map(p=>p.pts),
                        feather_um:+g('feather').value};}
g('preview').onclick=async()=>{
  if(!spec().keep.length){g('msg').textContent='draw a KEEP polygon first';return;}
  g('msg').textContent='rendering...';
  const r=await fetch('/preview',{method:'POST',body:JSON.stringify(spec())});
  if(!r.ok){g('msg').textContent='error: '+await r.text();return;}
  const b=await r.blob(); const im=new Image(); im.onload=()=>{shown=im;draw();g('msg').textContent='';};
  im.src=URL.createObjectURL(b);};
g('show').onclick=()=>{shown=base;draw();};
g('save').onclick=async()=>{
  if(!spec().keep.length){g('msg').textContent='draw a KEEP polygon first';return;}
  const r=await fetch('/save',{method:'POST',body:JSON.stringify(spec())});
  g('msg').textContent=(r.ok?'':'error: ')+await r.text();};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        a = STATE["a"]
        if u.path in ("/", "/index.html"):
            self._send(PAGE.replace("__W__", str(a.width))
                       .replace("__H__", str(a.height)).encode("utf-8"))
        elif u.path == "/base.png":
            with LOCK:
                self._send(render(), "image/png")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        spec = json.loads(self.rfile.read(n) or b"{}")
        for kind in ("keep", "remove"):
            for poly in spec.get(kind, []):
                if any(x is None or y is None for x, y in poly):
                    self.send_error(400, "polygon has undefined vertices")
                    return
        with LOCK:
            m, keep_full, remove_full = rasterise(spec, float(spec.get("feather_um", 0)))
            if self.path == "/preview":
                self._send(render(STATE["xp"].asarray(m)), "image/png")
                return
            if self.path == "/save":
                a = STATE["a"]
                out = Path(a.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                doc = dict(bin=STATE["cmeta"].get("bin_xy"),
                           cache_dir=a.cache_dir, frame_drawn_on=STATE["t"],
                           keep=keep_full, remove=remove_full,
                           feather_um=float(spec.get("feather_um", 0)),
                           view=dict(roll=a.roll, zoom=a.zoom, pan_x=a.pan_x,
                                     pan_y=a.pan_y, width=a.width, height=a.height,
                                     crop=list(STATE["crop"]) if STATE["crop"] else None))
                out.write_text(json.dumps(doc, indent=1))
                np.save(out.with_suffix(".mask.npy"), m)
                msg = (f"saved {out}\nkeeps {100 * m.mean():.0f}% of the crop\n\n"
                       f"add to the render command:\n  --roi {out}")
                self._send(msg.encode("utf-8"), "text/plain; charset=utf-8")
                return
        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--bin", type=int, default=2)
    p.add_argument("--signal-channel", type=int, default=0)
    p.add_argument("--baseline-n", type=int, default=5)
    p.add_argument("--bg-sigma", type=float, default=24.0)
    p.add_argument("--bg-mode", choices=["image", "plane-const"], default="plane-const")
    p.add_argument("--bg-pct", type=float, default=50.0)
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--crop-frac", type=float, default=0.3)
    p.add_argument("--crop-pad-um", type=float, default=60.0)
    p.add_argument("--frame", type=int, default=-1, help="timepoint to draw on (-1 = last)")
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--pan-x", type=float, default=0.0)
    p.add_argument("--pan-y", type=float, default=0.0)
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=620)
    p.add_argument("--vmin", type=float, default=8.0)
    p.add_argument("--vmax", type=float, default=72.0)
    p.add_argument("--emit", type=float, default=0.65)
    p.add_argument("--opacity", type=float, default=0.3)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--mode", choices=["composite", "mip"], default="composite")
    p.add_argument("--cmap", default="green")
    p.add_argument("--out", required=True, help="roi.json to write")
    p.add_argument("--port", type=int, default=8780)
    a = p.parse_args()
    setup(a)

    class DualStack(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        daemon_threads = True

        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
            ThreadingHTTPServer.server_bind(self)

    try:
        srv = DualStack(("::", a.port), Handler)
    except OSError:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"\n[roi] open  http://127.0.0.1:{a.port}   -- draw, Preview, Save\n"
          f"[roi] Ctrl-C to stop")
    srv.serve_forever()


if __name__ == "__main__":
    main()
