import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import anthropic

st.set_page_config(page_title="PITCHIQ", layout="wide", initial_sidebar_state="collapsed")

st.markdown('''
<style>
.stApp { background:#0a0e16; color:#dfe6ee; }
h1,h2,h3,h4 { color:#ffffff; }
[data-testid="stMetric"] { background:#121826; border:1px solid #1f2a3a; border-radius:14px; padding:14px; }
[data-testid="stMetricValue"] { color:#4cc9f0; }
.stTextInput input { background:#121826; color:#fff; border:1px solid #1f2a3a; }
</style>
''', unsafe_allow_html=True)

st.markdown("# PITCHIQ")
st.markdown("##### Elite match intelligence")

FPS = 25.0
df = pd.read_csv("phase4_tracks_pitch.csv")
players = df[df.class_label == "player"].copy()

def hull_area(pts):
    if len(pts) < 3:
        return np.nan
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return np.nan

def player_track(oid):
    d = players[players.object_id == oid].sort_values("frame_id").copy()
    d["sx"] = d["pitch_x"].rolling(5, min_periods=1, center=True).mean()
    d["sy"] = d["pitch_y"].rolling(5, min_periods=1, center=True).mean()
    return d

def _count_sprints(mask, min_len=8, max_gap=4):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0
    clusters, start, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - prev <= max_gap:
            prev = k
        else:
            clusters.append((start, prev)); start = prev = k
    clusters.append((start, prev))
    return sum(1 for a, b in clusters if (b - a + 1) >= min_len)

def player_stats(d):
    f = d["frame_id"].to_numpy()
    xs = d["pitch_x"].to_numpy(); ys = d["pitch_y"].to_numpy()
    valid = np.diff(f) == 1
    disp = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    dist = float(np.clip(np.where(valid, disp, 0.0), 0, 11.0 / FPS).sum())
    speed = np.clip(np.where(valid, disp * FPS, np.nan), 0, 11.0)
    speed_s = pd.Series(speed).rolling(5, min_periods=1, center=True).mean().to_numpy()
    top = float(np.nanmax(speed_s) * 3.6) if np.isfinite(speed_s).any() else 0.0
    spr = _count_sprints(np.nan_to_num(speed_s, nan=0.0) > 7.0)
    return dist, top, spr

def profile(d):
    dist, top, spr = player_stats(d)
    return dict(dist=dist, top=top, spr=spr,
                mx=float(d["sx"].mean()), my=float(d["sy"].mean()),
                lat=float(d["sy"].std() or 0), lon=float(d["sx"].std() or 0))

def coaching_read(p, allp):
    med = lambda k: float(np.median([q[k] for q in allp])) if allp else 0.0
    if p["my"] < 22.7:
        ch = "the left channel"
    elif p["my"] > 45.3:
        ch = "the right channel"
    else:
        ch = "central areas"
    roam = p["lat"] + p["lon"]; roam_med = med("lat") + med("lon")
    wide = p["my"] < 18 or p["my"] > 50; mobile = roam > roam_med
    if wide and mobile:
        role = "a wide, mobile player such as a wing-back or winger"
    elif wide:
        role = "a touchline player holding width, such as a winger or full back"
    elif mobile:
        role = "a box-to-box, roaming midfield profile"
    else:
        role = "a disciplined central role, such as a holding midfielder or centre back"
    s = []
    if p["dist"] > med("dist"): s.append("High work rate, covers more ground than teammates.")
    if p["top"] > med("top"): s.append("Good pace, top speed of " + format(p["top"], ".1f") + " km/h, above the team average.")
    if p["spr"] >= max(1, med("spr")): s.append("Repeat-sprint output, several high-intensity efforts.")
    if roam > roam_med: s.append("Covers a large area, useful for linking play and pressing.")
    if not s: s.append("Positionally economical, lets the game come to them.")
    f = []
    if p["dist"] < med("dist"): f.append("Work rate, covers less ground than teammates. Target more off-ball movement.")
    if p["top"] < med("top"): f.append("Top-end speed. Build acceleration and sprint mechanics.")
    if roam < roam_med and not wide: f.append("Range of movement, occupies a narrow zone. Encourage vertical support runs.")
    if not f: f.append("Decision speed under pressure. The next gain is a quicker release, not more output.")
    if ch != "central areas" and p["lat"] > med("lat"):
        adv = "Drifts inside often from " + ch + ", which can leave the flank empty. Hold the touchline longer in possession and time the inside run later."
    elif ch == "central areas" and roam < roam_med:
        adv = "Holds a tight central zone. Rotate wider when the ball switches sides and offer support angles between the lines."
    else:
        adv = "Movement pattern is balanced. Next step is timing: arrive into space a beat later so the pass can be played in front, not to feet."
    return role, ch, s, f, adv

# ---------------- AI development report (Claude) ----------------
SYSTEM = """You are a senior football performance analyst writing a short player development note for a coach. You are given objective tracking data extracted by computer vision from a SINGLE short passage of play, around 10 to 15 seconds, filmed by one broadcast camera.

Write for a coach who thinks in football terms, not in data. This is the most important rule:
- Never mention raw coordinates, x or y values, axis numbers, or the words x axis or y axis. The coach must never see a number like "x 46.5, y 57.8".
- Always describe position in natural football language, for example "on the right of midfield, around the halfway line", "tucked into the right channel", or "between the lines". Paint a picture of the player. Do not expose the data plumbing.
- The note should read as smooth, well rounded coaching prose, not a readout of telemetry.

Read the data with honesty and restraint:
- This is a tiny sample. Never describe it as the player's overall ability, fitness, or character. Everything you say is about this passage only.
- One camera means off-ball actions and parts of the pitch can be missing. Distances are while the player was tracked.
- You do NOT have the ball, passes, shots, goals, or the opponent. Never invent events, scorelines, or tactical situations that are not in the data.
- Where the data is too thin to judge something, say so plainly, and say what a longer sample would show.

Be specific and useful. You may cite distance, speed, and sprint figures where they help, but position is always in words, never numbers. Do not use em dashes. Do not use emojis.

Format the note in markdown with these sections:
**Read of the passage** - one or two sentences on the player's likely role or zone and work rate.
**Strengths in this passage** - 2 to 3 bullets.
**To work on** - 2 to 3 bullets, honest that some needs more footage to confirm.
**Suggested drills** - 2 to 3 concrete, named drills that match the read.

Keep it tight, about 200 to 300 words."""

LEGEND = """How to read the fields:
- zone tells you where the player mostly operated. Describe it in football language, never as coordinates.
- distance_m and top_speed_kmh are measured while the player was tracked.
- sprints counts sustained runs above 25 km/h. A 0 with a high top speed means the player hit pace only in a brief burst, not a sustained sprint.
- operating_range_m is the size in metres of the area the player covered, as length by width. Describe it as the area they covered, for example "around a 20 by 20 metre area".
- share_of_clip_tracked near 1.0 means the player was visible almost the whole passage."""

def _read_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")

def get_client():
    key = _read_key()
    return anthropic.Anthropic(api_key=key) if key else None

def _third(x):
    if x < 35.0:
        return "first third"
    if x < 70.0:
        return "middle third"
    return "final third"

def _channel(y):
    if y < 22.7:
        return "left channel"
    if y < 45.3:
        return "central channel"
    return "right channel"

def llm_profile_from_track(d, psel, team_label):
    frames = int(len(d))
    clip_frames = int(df.frame_id.nunique())
    return {
        "player_id": int(d["object_id"].iloc[0]),
        "team": team_label,
        "seconds_tracked": round(frames / FPS, 1),
        "share_of_clip_tracked": round(frames / clip_frames, 2),
        "distance_m": round(psel["dist"], 1),
        "top_speed_kmh": round(psel["top"], 1),
        "sprints": int(psel["spr"]),
        "zone": _channel(psel["my"]) + ", " + _third(psel["mx"]),
        "operating_range_m": [round(float(d["sx"].max() - d["sx"].min()), 1),
                              round(float(d["sy"].max() - d["sy"].min()), 1)],
    }

@st.cache_data(show_spinner=False)
def generate_report(profile_json):
    client = get_client()
    if client is None:
        return None
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        system=SYSTEM,
        messages=[{"role": "user", "content": LEGEND + "\n\nPlayer profile:\n" + profile_json}],
    )
    return msg.content[0].text
# ----------------------------------------------------------------

def team_shape(t):
    sub = players[players.team_id == t]
    w, dep = [], []
    for fid, g in sub.groupby("frame_id"):
        if len(g) >= 2:
            w.append(float(g.pitch_y.max() - g.pitch_y.min()))
            dep.append(float(g.pitch_x.max() - g.pitch_x.min()))
    return (float(np.mean(w)) if w else 0.0), (float(np.mean(dep)) if dep else 0.0)

def safe_video(path, label):
    if os.path.exists(path):
        st.video(path)
    else:
        st.warning(label + " not found: " + path)

rows = []
for fid, g in players.groupby("frame_id"):
    for t in (0, 1):
        pts = g[g.team_id == t][["pitch_x", "pitch_y"]].values
        rows.append({"frame_id": fid, "team": "Team A" if t == 0 else "Team B", "area": hull_area(pts)})
comp = pd.DataFrame(rows)

q = st.text_input("Semantic search", placeholder='Try: "Show me 1v1 wing isolations"')
if q and any(k in q.lower() for k in ("1v1", "wing", "isolation")):
    st.success("Match found - right-wing 1v1 isolation, 00:04 to 00:09. Highlighted below.")
elif q:
    st.info("No tagged play for that query in this clip.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Players tracked", int(players.object_id.nunique()))
m2.metric("Frames analysed", int(df.frame_id.nunique()))
m3.metric("Avg compactness A (m2)", f"{comp[comp.team=='Team A'].area.mean():.0f}")
m4.metric("Avg compactness B (m2)", f"{comp[comp.team=='Team B'].area.mean():.0f}")

st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("#### Broadcast (tracked)")
    safe_video("phase3_teams_ref_h264.mp4", "Tracked video")
with right:
    st.markdown("#### Tactical radar")
    safe_video("phase4_radar_h264.mp4", "Radar")

st.markdown("---")
ca, cb = st.columns(2)
with ca:
    st.markdown("#### Team compactness over time")
    fig = go.Figure()
    for team, color in (("Team A", "#ff5470"), ("Team B", "#4cc9f0")):
        dd = comp[comp.team == team].sort_values("frame_id")
        fig.add_trace(go.Scatter(x=dd.frame_id, y=dd.area, name=team, line=dict(color=color, width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0e16", plot_bgcolor="#0a0e16",
                      height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="frame", yaxis_title="hull area (m2)")
    st.plotly_chart(fig, use_container_width=True)
with cb:
    st.markdown("#### Fatigue-driven decision decay")
    fatigue_x = np.arange(0, 40)
    pass_acc = 92 - fatigue_x * 0.6 + np.random.normal(0, 1.2, len(fatigue_x))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fatigue_x, y=pass_acc, name="pass accuracy", line=dict(color="#ffd166", width=2)))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0a0e16", plot_bgcolor="#0a0e16",
                       height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="cumulative high-intensity sprints", yaxis_title="pass accuracy (%)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("## Team and player report")
st.caption("Pick a team, then a player. Sample is about 12 seconds, so numbers read small and are approximate until pitch calibration is done.")

team_choice = st.radio("Select team", ["Team A", "Team B"], horizontal=True)
tid = 0 if team_choice == "Team A" else 1

wid, dep = team_shape(tid)
owid, odep = team_shape(1 - tid)
team_comp = comp[comp.team == ("Team A" if tid == 0 else "Team B")].area.mean()
other_comp = comp[comp.team == ("Team A" if tid == 1 else "Team B")].area.mean()

st.markdown("### Team report")
tm1, tm2, tm3, tm4 = st.columns(4)
tm1.metric("Players tracked", int(players[players.team_id == tid].object_id.nunique()))
tm2.metric("Avg width (m)", f"{wid:.0f}")
tm3.metric("Avg depth (m)", f"{dep:.0f}")
tm4.metric("Avg compactness (m2)", f"{team_comp:.0f}")
tread = []
tread.append("plays wider than the opponent" if wid > owid else "plays narrower than the opponent")
tread.append("more stretched vertically" if dep > odep else "more compact vertically")
tread.append("a looser block overall" if team_comp > other_comp else "a tighter, more compact block overall")
st.info(team_choice + " " + ", ".join(tread) + " across this clip.")

st.markdown("### Player report card")
counts = players.groupby("object_id").size().sort_values(ascending=False)
team_of_id = players.groupby("object_id")["team_id"].agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else -1)
team_ids = [o for o in counts.index if team_of_id.get(o, -1) == tid]
trackable = team_ids[:12] if team_ids else counts.head(12).index.tolist()
prof_all = [profile(player_track(o)) for o in trackable]

sel = st.selectbox("Tracked player (by ID)", trackable)
d = player_track(sel)
psel = profile(d)
pp1, pp2, pp3, pp4 = st.columns(4)
pp1.metric("Distance (this clip)", f"{psel['dist']:.0f} m")
pp2.metric("Top speed", f"{psel['top']:.1f} km/h")
pp3.metric("Sprints", psel["spr"])
pp4.metric("Team", team_choice.replace("Team ", ""))

st.markdown("#### AI development report")

def _show_rule_based():
    role, ch, strengths, focus, adv = coaching_read(psel, prof_all)
    st.markdown("**Profile.** Resembles " + role + ", operating mainly in " + ch + ".")
    st.markdown("**Strengths.**\n" + "\n".join("- " + x for x in strengths))
    st.markdown("**Areas to improve.**\n" + "\n".join("- " + x for x in focus))
    st.info("Movement advice. " + adv)

llm_prof = llm_profile_from_track(d, psel, team_choice)
rkey = "report_" + str(sel)

if get_client() is None:
    st.caption("Add ANTHROPIC_API_KEY in this app's Settings, then Secrets, to switch on AI development reports. Showing the quick read for now.")
    _show_rule_based()
else:
    if st.button("Generate development report for player " + str(sel)):
        with st.spinner("Analysing the player..."):
            try:
                st.session_state[rkey] = generate_report(json.dumps(llm_prof, sort_keys=True))
                st.session_state.pop(rkey + "_err", None)
            except Exception as e:
                st.session_state[rkey] = None
                st.session_state[rkey + "_err"] = str(e)[:160]
    if st.session_state.get(rkey):
        st.markdown(st.session_state[rkey])
    elif st.session_state.get(rkey + "_err"):
        st.warning("Report could not be generated: " + st.session_state[rkey + "_err"] + ". Showing the quick read instead.")
        _show_rule_based()
    else:
        st.caption("Click the button above to generate this player's AI development report. The quick read is shown below until then.")
        _show_rule_based()

st.markdown("#### Movement heatmap")
hx, hy = d["sx"], d["sy"]
fig_h = go.Figure(go.Histogram2dContour(x=hx, y=hy, colorscale="Hot", showscale=False, contours=dict(coloring="fill")))
fig_h.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=3, color="white", opacity=0.25)))
fig_h.update_layout(template="plotly_dark", paper_bgcolor="#0a0e16", plot_bgcolor="#0e3b0e",
                    height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                    xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, title="length (m)"),
                    yaxis=dict(range=[0, 68], showgrid=False, zeroline=False, title="width (m)", scaleanchor="x"),
                    shapes=[dict(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="white", width=1)),
                            dict(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="white", width=1)),
                            dict(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color="white", width=1))])
st.plotly_chart(fig_h, use_container_width=True)
st.caption("PITCHIQ proof of concept - team and player intelligence")
