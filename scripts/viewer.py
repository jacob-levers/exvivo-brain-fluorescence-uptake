#!/usr/bin/env python
"""Interactive viewer: set the frame yourself, then copy the render command.

Starts a small local web server that renders frames on demand with the same
raycaster the movie uses, so what you see here is exactly what you will get.
Nothing is uploaded -- it reads the local cache and renders on the local GPU.

    python scripts/viewer.py --cache-dir out3 --bin 2 --port 8770

then open http://localhost:8770

Every control maps 1:1 to a flag of nd2_perfusion_3d.py, and the page shows the
full command for the current settings so you can render the movie from it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd2_perfusion_3d as P


class Args:
    """Stand-in for the CLI namespace that the pipeline helpers expect."""
    pass


STATE = {}
LOCK = threading.Lock()


def setup(a):
    args = Args()
    args.outdir = a.cache_dir
    args.cache_dir = a.cache_dir
    args.bin = a.bin
    args.gpu = True
    args.baseline_n = a.baseline_n
    args.bg_sigma = a.bg_sigma
    args.force_bg = False
    args.bg_mode = a.bg_mode
    args.bg_pct = a.bg_pct
    args.signal_channel = a.signal_channel
    args.crop_pad_um = a.crop_pad_um

    arr, cmeta = P.load_cache(args)
    bg = P.compute_background(arr, cmeta, args)
    xp, ndi, _ = P.backend(True)

    nT, nC, nZ, Yb, Xb = arr.shape
    shifts = np.zeros((nT, 3))
    sp = Path(a.cache_dir) / "shifts.csv"
    if sp.exists() and not a.no_register:
        tab = np.genfromtxt(sp, delimiter=",", names=True)
        shifts = np.c_[tab["shift_z_px"], tab["shift_y_px"], tab["shift_x_px"]]

    crop = None
    if a.crop:
        crop = P.sample_bbox(arr, bg, a.signal_channel, args, cmeta,
                             pad_um=a.crop_pad_um, frac=a.crop_frac)
        Yb, Xb = crop[1] - crop[0], crop[3] - crop[2]

    ts = (np.asarray(cmeta["times_s"]) if cmeta.get("times_s")
          else np.arange(nT, dtype=float))

    STATE.update(arr=arr, cmeta=cmeta, bg=bg, xp=xp, ndi=ndi, args=args,
                 shifts=shifts, crop=crop, nT=nT, nZ=nZ, Yb=Yb, Xb=Xb, ts=ts,
                 vol_cache={}, cli=a, nZfull=nZ)
    print(f"[viewer] cache {a.cache_dir} bin{a.bin}: {nT} timepoints, "
          f"{nZ} z-planes")
    print(f"[viewer] rendered volume {Xb} x {Yb} px = "
          f"{Xb*cmeta['dx_um']:.0f} x {Yb*cmeta['dy_um']:.0f} um, "
          f"{(nZ-1)*cmeta['dz_um']:.0f} um deep")
    print(f"[viewer] registration: "
          f"{'NOT applied' if a.no_register else 'applied from shifts.csv'}")


def get_volume(t, zr):
    key = (t, zr)
    vc = STATE["vol_cache"]
    if key not in vc:
        if len(vc) > 6:
            vc.clear()
        vc[key] = P.build_volume(
            STATE["arr"], STATE["bg"], STATE["shifts"], t,
            [STATE["cli"].signal_channel], STATE["xp"], STATE["ndi"],
            crop=STATE["crop"], zrange=zr)[0]
    return vc[key]


def get_mip(t, zr):
    """Max projection, kept on the host per timepoint.

    Recomputing this per request costs a full device-to-host copy of the
    volume, which dominated the response time -- far more than the raycast.
    """
    key = (t, zr)
    mc = STATE.setdefault("mip_cache", {})
    if key not in mc:
        if len(mc) > 12:
            mc.clear()
        mc[key] = P.to_host(get_volume(t, zr).max(axis=0))
    return mc[key]


def get_lut(name, xp):
    lc = STATE.setdefault("lut_cache", {})
    if name not in lc:
        host = P.make_lut(name)
        lc[name] = (host, xp.asarray(host))
    return lc[name]


def render_png(q):
    xp, ndi = STATE["xp"], STATE["ndi"]
    cm = STATE["cmeta"]

    def num(key, default):
        try:
            return float(q.get(key, [default])[0])
        except (TypeError, ValueError):
            return float(default)

    t = int(num("t", 0))
    t = max(0, min(STATE["nT"] - 1, t))
    W, H = int(num("w", 900)), int(num("h", 600))
    el, spin, az = num("el", 0), num("spin", 0), num("az", 0)
    zoom = num("zoom", 1.0)
    vmin, vmax = num("vmin", 8), num("vmax", 34)
    emit, opac = num("emit", 0.9), num("opac", 0.35)
    samples = int(num("samples", 320))
    cut = q.get("cut", ["0"])[0] == "1"
    span, yaw = num("span", 45), num("yaw", 0)
    cmap = q.get("cmap", ["inferno"])[0]
    roll = num("roll", 0)
    panx, pany = num("panx", 0), num("pany", 0)
    cutdx, cutdy = num("cutdx", 0), num("cutdy", 0)
    zmin = int(num("zmin", 0))
    zmax = int(num("zmax", STATE["nZfull"]))
    zmin = max(0, min(STATE["nZfull"] - 1, zmin))
    zmax = max(zmin + 1, min(STATE["nZfull"], zmax))

    vol = get_volume(t, (zmin, zmax))
    rc = P.Raycaster((zmax - zmin, STATE["Yb"], STATE["Xb"]),
                     (cm["dz_um"], cm["dy_um"], cm["dx_um"]), W, H, xp, ndi)
    lut_host, lut = get_lut(cmap, xp)

    cc = (0.0, 0.0)
    if cut:
        m = get_mip(t, (zmin, zmax))
        w = np.clip(m - vmin, 0, None)
        if w.sum() > 0:
            yc = float((w.sum(1) * np.arange(STATE["Yb"])).sum() / w.sum())
            xc = float((w.sum(0) * np.arange(STATE["Xb"])).sum() / w.sum())
            cc = (xc * cm["dx_um"] - rc.Lx / 2, yc * cm["dy_um"] - rc.Ly / 2)
    cc = (cc[0] + cutdx, cc[1] + cutdy)

    img = rc.render([vol], [lut], [vmin], [vmax], [opac], az, el,
                    n_samples=samples, zoom=zoom, emit=[emit],
                    cutaway=cut, cut_center=cc, spin=spin,
                    cut_span=span, cut_yaw=yaw, roll=roll, pan=(panx, pany))
    img8 = (P.to_host(img) * 255).astype(np.uint8)

    sig = cm["channels"][STATE["cli"].signal_channel]
    img8 = P.annotate(
        img8, elapsed_s=float(STATE["ts"][t]),
        um_per_screen_px=rc.um_per_screen_px(zoom),
        label=f"c{STATE['cli'].signal_channel} {sig['name']}  |  "
              f"fixed scale [{vmin:.0f},{vmax:.0f}]",
        transit=False,
        cbar=(lut_host, vmin, vmax, sig["name"]),
        scalebar_um=200.0)

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>volume viewer</title><style>
body{background:#111;color:#ddd;font:13px system-ui,sans-serif;margin:0;
     display:flex;height:100vh}
#side{width:290px;padding:14px;overflow-y:auto;background:#181818;
      border-right:1px solid #2c2c2c}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:10px}
img{max-width:100%;max-height:92vh;image-rendering:auto}
label{display:block;margin:9px 0 2px;color:#9a9a9a;font-size:11px;
      text-transform:uppercase;letter-spacing:.4px}
input[type=range]{width:100%}
.row{display:flex;gap:6px}.row>*{flex:1}
input[type=number],select{background:#222;color:#ddd;border:1px solid #383838;
      border-radius:3px;padding:3px 5px;width:100%}
.v{float:right;color:#e8a33d;font-weight:600}
button{background:#2a2a2a;color:#ddd;border:1px solid #3d3d3d;border-radius:4px;
       padding:6px 9px;cursor:pointer;width:100%;margin-top:10px}
button:hover{background:#333}
#cmd{margin-top:8px;background:#0d0d0d;border:1px solid #2c2c2c;padding:8px;
     font:11px ui-monospace,monospace;color:#7fc98a;white-space:pre-wrap;
     word-break:break-all;max-height:190px;overflow-y:auto}
hr{border:0;border-top:1px solid #2c2c2c;margin:14px 0}
</style></head><body>
<div id="side">
<label>Timepoint <span class="v" id="tv">0</span></label>
<input type="range" id="t" min="0" max="__NT__" value="0">
<hr>
<label>Elevation <span class="v" id="elv">0</span>&deg;</label>
<input type="range" id="el" min="0" max="90" step="1" value="0">
<label>Spin (turntable) <span class="v" id="spinv">0</span>&deg;</label>
<input type="range" id="spin" min="0" max="360" step="1" value="0">
<label>Azimuth <span class="v" id="azv">0</span>&deg;</label>
<input type="range" id="az" min="-90" max="90" step="1" value="0">
<label>Roll (turn in frame) <span class="v" id="rollv">0</span>&deg;</label>
<input type="range" id="roll" min="-180" max="180" step="1" value="0">
<label>Zoom <span class="v" id="zoomv">1.00</span></label>
<input type="range" id="zoom" min="0.25" max="4" step="0.01" value="1">
<label>Pan X <span class="v" id="panxv">0</span> um</label>
<input type="range" id="panx" min="-900" max="900" step="5" value="0">
<label>Pan Y <span class="v" id="panyv">0</span> um</label>
<input type="range" id="pany" min="-900" max="900" step="5" value="0">
<button id="reset">Reset position</button>
<hr>
<label>Z slab: first / last plane</label>
<div class="row"><input type="number" id="zmin" value="0" step="1" min="0">
<input type="number" id="zmax" value="__NZ__" step="1" min="1"></div>
<hr>
<label><input type="checkbox" id="cut"> quarter cutaway</label>
<label>Cut span <span class="v" id="spanv">45</span>&deg; (90 = half)</label>
<input type="range" id="span" min="10" max="90" step="1" value="45">
<label>Cut yaw <span class="v" id="yawv">0</span>&deg;</label>
<input type="range" id="yaw" min="-90" max="90" step="1" value="0">
<label>Cut offset X <span class="v" id="cutdxv">0</span> um</label>
<input type="range" id="cutdx" min="-500" max="500" step="5" value="0">
<label>Cut offset Y <span class="v" id="cutdyv">0</span> um</label>
<input type="range" id="cutdy" min="-500" max="500" step="5" value="0">
<hr>
<label>Display min / max</label>
<div class="row"><input type="number" id="vmin" value="8" step="1">
<input type="number" id="vmax" value="34" step="1"></div>
<label>Emission <span class="v" id="emitv">0.90</span></label>
<input type="range" id="emit" min="0.1" max="4" step="0.05" value="0.9">
<label>Absorption <span class="v" id="opacv">0.35</span></label>
<input type="range" id="opac" min="0.02" max="3" step="0.01" value="0.35">
<hr>
<label>Frame width / height</label>
<div class="row"><input type="number" id="w" value="900" step="20">
<input type="number" id="h" value="600" step="20"></div>
<label>Colormap</label>
<select id="cmap"><option>inferno</option><option>magma</option>
<option>viridis</option><option>hot</option><option>gray</option>
<option>turbo</option></select>
<label>Ray samples <span class="v" id="samplesv">320</span></label>
<input type="range" id="samples" min="80" max="800" step="20" value="320">
<button id="copy">Copy render command</button>
<div id="cmd"></div>
</div>
<div id="main"><img id="view" src=""></div>
<script>
const ids=["t","el","spin","az","roll","zoom","panx","pany","zmin","zmax",
           "cut","span","yaw","cutdx","cutdy","vmin","vmax",
           "emit","opac","w","h","cmap","samples"];
const g=i=>document.getElementById(i);
function vals(){const o={};for(const i of ids){const e=g(i);
  o[i]=e.type==="checkbox"?(e.checked?1:0):e.value;}return o;}
function fmt(){const v=vals();
  for(const [k,d] of [["t","tv"],["el","elv"],["spin","spinv"],["az","azv"],
      ["span","spanv"],["yaw","yawv"],["samples","samplesv"],["roll","rollv"],
      ["panx","panxv"],["pany","panyv"],["cutdx","cutdxv"],["cutdy","cutdyv"]])
    g(d).textContent=v[k];
  g("zoomv").textContent=(+v.zoom).toFixed(2);
  g("emitv").textContent=(+v.emit).toFixed(2);
  g("opacv").textContent=(+v.opac).toFixed(2);}
let pending=false,queued=false;
function draw(){
  if(pending){queued=true;return;}
  pending=true;const v=vals();
  const q=new URLSearchParams(v).toString();
  const im=new Image();
  im.onload=()=>{g("view").src=im.src;pending=false;
                 if(queued){queued=false;draw();}};
  im.onerror=()=>{pending=false;};
  im.src="/render?"+q;
  cmd();
}
function cmd(){const v=vals();
  let s="python scripts/nd2_perfusion_3d.py render \\\n"+
    "  --nd2 \"__ND2__\" --outdir __OUT__ --cache-dir __OUT__ --bin __BIN__ \\\n"+
    "  __BGFLAGS__ __REGFLAG__ __CROPFLAG__ \\\n"+
    "  --vmin "+v.vmin+" --vmax "+v.vmax+
    " --emit "+v.emit+" --opacity "+v.opac+" \\\n"+
    "  --azimuth "+v.az+" --elevation "+v.el+
    " --roll "+v.roll+" --zoom "+v.zoom+" \\\n"+
    "  --pan-x "+v.panx+" --pan-y "+v.pany+" \\\n"+
    "  --width "+v.w+" --height "+v.h+" --samples "+v.samples+
    " --cmap "+v.cmap+" \\\n";
  if(v.zmin!="0"||v.zmax!=__NZ__) s+="  --z-min "+v.zmin+" --z-max "+v.zmax+" \\\n";
  if(v.cut==1) s+="  --cutaway --cut-span "+v.span+" --cut-yaw "+v.yaw+
                  " --cut-dx "+v.cutdx+" --cut-dy "+v.cutdy+" \\\n";
  s+="  --fps 12 --out __OUT__/my_view.mp4";
  g("cmd").textContent=s;}
for(const i of ids){const e=g(i);
  e.addEventListener("input",()=>{fmt();draw();});
  e.addEventListener("change",()=>{fmt();draw();});}
g("copy").onclick=()=>{navigator.clipboard.writeText(g("cmd").textContent);
  g("copy").textContent="Copied";
  setTimeout(()=>g("copy").textContent="Copy render command",1200);};

// ---- mouse: drag orbits, shift/right-drag pans, wheel zooms ----
const view=g("view");
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const bump=(id,d,a,b)=>{g(id).value=clamp(parseFloat(g(id).value)+d,a,b);};
let drag=null;
view.addEventListener("contextmenu",e=>e.preventDefault());
view.addEventListener("mousedown",e=>{
  drag={x:e.clientX,y:e.clientY,pan:(e.shiftKey||e.button===2)};
  e.preventDefault();});
window.addEventListener("mouseup",()=>{drag=null;});
window.addEventListener("mousemove",e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){
    // pan in microns: scale by how many microns a screen pixel covers
    const k=2.2/parseFloat(g("zoom").value);
    bump("panx",-dx*k,-900,900); bump("pany",dy*k,-900,900);
  }else{
    bump("spin",dx*0.5,0,360); bump("el",-dy*0.4,0,90);
  }
  fmt();draw();});
view.addEventListener("wheel",e=>{
  e.preventDefault();
  const z=parseFloat(g("zoom").value)*(e.deltaY>0?0.94:1.064);
  g("zoom").value=clamp(z,0.25,4); fmt();draw();},{passive:false});
g("reset").onclick=()=>{
  for(const [id,val] of [["el",0],["spin",0],["az",0],["roll",0],["zoom",1],
                         ["panx",0],["pany",0],["cutdx",0],["cutdy",0]])
    g(id).value=val;
  fmt();draw();};
fmt();draw();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            c = STATE["cli"]
            bgflags = f"--bg-mode {c.bg_mode} --bg-pct {c.bg_pct:g}"
            if c.bg_mode == "image":
                bgflags = f"--bg-mode image --bg-sigma {c.bg_sigma:g}"
            page = (PAGE
                    .replace("__NT__", str(STATE["nT"] - 1))
                    .replace("__NZ__", str(STATE["nZfull"]))
                    .replace("__ND2__", c.nd2 or "YOUR.nd2")
                    .replace("__OUT__", c.cache_dir)
                    .replace("__BIN__", str(c.bin))
                    .replace("__BGFLAGS__", bgflags)
                    .replace("__REGFLAG__",
                             "--no-register" if c.no_register else "")
                    .replace("__CROPFLAG__",
                             f"--crop --crop-frac {c.crop_frac:g}"
                             if c.crop else ""))
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/render":
            q = urllib.parse.parse_qs(u.query)
            try:
                with LOCK:
                    png = render_png(q)
            except Exception as e:                     # pragma: no cover
                self.send_error(500, str(e))
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            return
        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nd2", default=None, help="only used to fill in the command text")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--bin", type=int, default=2)
    p.add_argument("--signal-channel", type=int, default=0)
    p.add_argument("--baseline-n", type=int, default=5)
    p.add_argument("--bg-sigma", type=float, default=24.0)
    p.add_argument("--bg-mode", choices=["image", "plane-const"],
                   default="plane-const")
    p.add_argument("--bg-pct", type=float, default=50.0)
    p.add_argument("--no-register", action="store_true", default=True)
    p.add_argument("--register", dest="no_register", action="store_false")
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--crop-frac", type=float, default=0.5)
    p.add_argument("--crop-pad-um", type=float, default=60.0)
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()

    setup(a)

    # Dual-stack, deliberately.  Bound to IPv4 only, a browser asking for
    # "localhost" tries ::1 first and waits out a ~2 s failure before falling
    # back -- which swamps the 0.3 s render and makes the viewer feel broken.
    class DualStack(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        daemon_threads = True

        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6,
                                       socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
            ThreadingHTTPServer.server_bind(self)

    try:
        srv = DualStack(("::", a.port), Handler)
    except OSError:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)

    print(f"\n[viewer] open  http://127.0.0.1:{a.port}"
          f"   (or http://localhost:{a.port})\n"
          f"[viewer] ~0.3 s per frame; Ctrl-C to stop")
    srv.serve_forever()


if __name__ == "__main__":
    main()
