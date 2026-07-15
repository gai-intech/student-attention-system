"""
app.py — Gradio dashboard for the Student Attention System.
Run:  python src/app.py   then open http://127.0.0.1:7860
"""

import os, time, tempfile
import numpy as np
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import AttentionPipeline

pipe = AttentionPipeline(detector_conf=0.30, imgsz=640)

# live session state (event log)
LIVE = {"start": None, "last_state": None, "events": []}


# ============================================================================
# LIVE dashboard helpers
# ============================================================================
def _state(att_pct):
    if att_pct >= 60:  return "ENGAGED", "#1B8A3A", "#E7F6EC"
    if att_pct >= 35:  return "PARTIAL", "#C77800", "#FFF4E0"
    return "LOOKING AWAY", "#B00020", "#FDE7EA"

def _bar(pct, color):
    pct = max(0, min(100, pct))
    return (f"<div style='background:#e2e2e2;border-radius:8px;height:16px;width:100%;overflow:hidden'>"
            f"<div style='background:{color};height:16px;width:{pct:.0f}%'></div></div>")

def _card(title, color, bg, lines, pct):
    body = "".join(
        f"<div style='font-size:15px;color:#111;margin:3px 0'>{ln}</div>" for ln in lines)
    return f"""<div style='font-family:system-ui;border-radius:14px;padding:16px 20px;
        background:{bg};border:2px solid {color}'>
      <div style='font-size:26px;font-weight:800;color:{color};letter-spacing:1px'>{title}</div>
      <div style='margin:10px 0'>{_bar(pct,color)}</div>{body}</div>"""

def _idle_card():
    return _card("WAITING", "#555", "#F2F2F2",
                 ["No person detected yet — move into the camera view."], 0)

def _events_html():
    if not LIVE["events"]:
        return ("<div style='font-family:system-ui;color:#888;padding:10px'>"
                "Event log — state changes will appear here as they happen.</div>")
    rows = []
    for t, state in reversed(LIVE["events"][-15:]):
        c = {"ENGAGED":"#1B8A3A","PARTIAL":"#C77800","LOOKING AWAY":"#B00020",
             "NO PERSON":"#555"}.get(state,"#333")
        mm, ss = divmod(int(t), 60)
        rows.append(f"<tr><td style='padding:4px 10px;color:#555'>{mm:02d}:{ss:02d}</td>"
                    f"<td style='padding:4px 10px;font-weight:700;color:{c}'>{state}</td></tr>")
    return ("<div style='font-family:system-ui'><b>Event log</b>"
            "<table style='width:100%;border-collapse:collapse;margin-top:6px'>"
            "<tr style='color:#888;font-size:13px'><td style='padding:4px 10px'>time</td>"
            "<td style='padding:4px 10px'>state</td></tr>" + "".join(rows) + "</table></div>")

def _live_card(people, mode):
    if not people:
        cur = "NO PERSON"; card = _idle_card()
    elif mode != "Full class":
        p = max(people, key=lambda q:(q["box"][2]-q["box"][0])*(q["box"][3]-q["box"][1]))
        att = p["attention"]*100
        title, color, bg = _state(att); cur = title
        card = _card(title, color, bg,
                     [f"<b>Behavior:</b> {p['behavior']}",
                      f"<b>Engagement:</b> {p['emotion']}",
                      f"<b>Attention:</b> {att:.0f}%"], att)
    else:
        atts=[q["attention"] for q in people]; avg=float(np.mean(atts))*100
        title,color,bg=_state(avg); cur=title
        eng=sum(1 for a in atts if a>=0.6)
        card=_card(f"CLASS: {title}", color, bg,
                   [f"<b>People detected:</b> {len(people)}",
                    f"<b>Engaged now:</b> {eng}/{len(people)}",
                    f"<b>Class attention:</b> {avg:.0f}%"], avg)
    return card, cur

def live_stream(frame_rgb, mode):
    if frame_rgb is None:
        return None, _idle_card(), _events_html()
    if LIVE["start"] is None:
        LIVE["start"] = time.time()
    t = time.time() - LIVE["start"]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, people = pipe.process_frame(frame_bgr, personal_mode=(mode!="Full class"))
    card, cur = _live_card(people, mode)
    # log a state CHANGE as an event
    if cur != LIVE["last_state"]:
        LIVE["events"].append((t, cur))
        LIVE["last_state"] = cur
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), card, _events_html()

def reset_session():
    pipe.reset()
    LIVE["start"] = None; LIVE["last_state"] = None; LIVE["events"] = []
    return _idle_card(), _events_html()


# ============================================================================
# UPLOAD dashboard helpers
# ============================================================================
def _curve_fig(curve):
    fig, ax = plt.subplots(figsize=(9,3))
    if curve and len(curve) >= 2:
        ts=[t for t,_ in curve]; ys=[100*a for _,a in curve]
        ax.plot(ts,ys,color="#1B5E20",lw=2); ax.fill_between(ts,ys,alpha=0.15,color="#1B5E20")
    else:
        ax.text(0.5,0.5,"Not enough frames for a curve",ha="center",va="center")
    ax.set_ylim(0,100); ax.set_xlabel("time (s)"); ax.set_ylabel("class attention %")
    ax.set_title("Class attention over time"); ax.grid(alpha=0.3); fig.tight_layout()
    return fig

def _behavior_fig(dist):
    fig, ax = plt.subplots(figsize=(9,3))
    if dist:
        items=sorted(dist.items(), key=lambda kv:-kv[1])
        ax.bar([k for k,_ in items],[v for _,v in items],color="#C9A96E")
        ax.set_ylabel("frames"); ax.tick_params(axis="x",rotation=30)
        for lb in ax.get_xticklabels(): lb.set_ha("right")
    else:
        ax.text(0.5,0.5,"No behavior data",ha="center",va="center")
    ax.set_title("Behavior distribution (whole video)"); fig.tight_layout()
    return fig

def _heatmap_fig(timeline, per_student):
    fig, ax = plt.subplots(figsize=(9,4))
    ids=[tid for tid,_ in sorted(per_student.items(), key=lambda kv:-kv[1]["attention_pct"])]
    ids=ids[:25]
    if not ids:
        ax.text(0.5,0.5,"No students",ha="center",va="center"); ax.axis("off"); return fig
    all_t=[t for tid in ids for t,_ in timeline.get(tid,[])]
    if not all_t:
        ax.text(0.5,0.5,"No timeline data",ha="center",va="center"); ax.axis("off"); return fig
    tmax=max(all_t); BINS=40
    mat=np.full((len(ids),BINS), np.nan)
    for r,tid in enumerate(ids):
        for t,a in timeline.get(tid,[]):
            b=min(BINS-1,int(t/max(tmax,1e-6)*BINS)); mat[r,b]=a
    im=ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"#{i}" for i in ids], fontsize=7)
    ax.set_xlabel("time  ->"); ax.set_title("Per-student attention over time (green=engaged, red=away)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); fig.tight_layout()
    return fig

def _report_md(rep):
    single = rep["student_count"] <= 1
    lines = ["### Personal Attention Analysis\n" if single else "### Class Attention Analysis\n",
             f"- **Students (estimate):** {rep['student_count']}   "
             f"(peak in one frame: {rep.get('peak_in_frame','-')}, unique tracks: {rep.get('unique_tracks','-')})"]
    if rep["teacher_id"] is not None:
        lines.append(f"- **Teacher identified:** track #{rep['teacher_id']} (excluded)")
    lines.append(f"- **Overall attention score:** **{rep['overall_attention']}%**")
    lines += ["\n**Per-student:**\n", "| Student | Attention % | Dominant behavior | frames |",
              "|---|---|---|---|"]
    for tid,s in sorted(rep["per_student"].items(), key=lambda kv:-kv[1]["attention_pct"]):
        lines.append(f"| #{tid} | {s['attention_pct']}% | {s['dominant_behavior']} | {s['frames']} |")
    return "\n".join(lines)

def analyze_upload(video_path, target_fps, mode, progress=gr.Progress()):
    if not video_path:
        return None, None, None, None, [], "", "Please upload a video."
    import imageio
    single = {"Auto":None,"Single person":True,"Full class":False}[mode]
    out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")
    writer = imageio.get_writer(out_path, fps=int(target_fps), codec="libx264",
                                quality=8, macro_block_size=None)
    def _prog(p): progress(p, desc="Analyzing video...")
    for frame in pipe.process_video(video_path, target_fps=int(target_fps),
                                    draw=True, progress=_prog, single_person=single):
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()
    rep = pipe.report()
    gallery = [(img, f"#{tid}  |  {ps['attention_pct']}%  |  {ps['dominant_behavior']}")
               for img, tid, ps in rep["crops"]]
    nl = chr(10)
    if rep["events"]:
        head = "### Movement events" + nl + nl + "| time | event |" + nl + "|---|---|" + nl
        body = nl.join(f"| {int(t//60):02d}:{int(t%60):02d} | {msg} |" for t, msg in rep["events"])
        ev = head + body
    else:
        ev = "### Movement events" + nl + nl + "No students entered or left during the clip."
    return (out_path, _curve_fig(rep["class_curve"]), _behavior_fig(rep["behavior_dist"]),
            _heatmap_fig(rep["timeline"], rep["per_student"]), gallery, ev, _report_md(rep))


# ============================================================================
# UI
# ============================================================================
with gr.Blocks(title="Student Attention System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Student Attention & Behavior Analysis")

    with gr.Tab("Live (real-time)"):
        gr.Markdown("Live webcam analysis. In **Personal / online class** mode it "
                    "focuses on one person; look away and the status flips within ~2s. "
                    "Every state change is recorded in the event log.")
        lmode = gr.Radio(["Personal / online class","Full class"],
                         value="Personal / online class", label="Mode")
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Camera", height=380)
            out = gr.Image(label="Live analysis", height=380)
        with gr.Row():
            dash = gr.HTML(_idle_card())
            events = gr.HTML(_events_html())
        reset_btn = gr.Button("Reset session")
        cam.stream(live_stream, inputs=[cam,lmode], outputs=[out,dash,events],
                   stream_every=0.3, show_progress="hidden")
        reset_btn.click(reset_session, outputs=[dash,events])

    with gr.Tab("Upload Video"):
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="Classroom / person video")
                fps = gr.Slider(1,5,value=2,step=1,label="Frames per second to analyze")
                mode = gr.Radio(["Auto","Single person","Full class"], value="Auto", label="Mode")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_vid = gr.Video(label="Annotated result")
        report = gr.Markdown()
        gallery = gr.Gallery(label="Detected students (crops with ID)", columns=6, height=260)
        events_md = gr.Markdown()
        with gr.Row():
            curve = gr.Plot(label="Attention over time")
            beh = gr.Plot(label="Behavior distribution")
        heat = gr.Plot(label="Per-student attention heatmap")
        btn.click(analyze_upload, [vid,fps,mode],
                  [out_vid,curve,beh,heat,gallery,events_md,report])


if __name__ == "__main__":
    demo.launch()
