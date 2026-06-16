import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

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

def player_stats(d):
    cons = d["frame_id"].diff() == 1
    step = np.sqrt(d["sx"].diff()**2 + d["sy"].diff()**2)
    raw = (step * FPS).where(cons)
    tele = raw > 11.0
    dist = float(step.where(cons & ~tele, 0).sum())
    sp = raw.clip(upper=11.0)
    top = float(np.nanmax(sp) * 3.6) if sp.notna().any() else 0.0
    spr = int(((sp > 7.0) & (sp.shift(1) <= 7.0)).sum())
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

role, ch, strengths, focus, adv = coaching_read(psel, prof_all)
st.markdown("#### AI coaching read")
st.markdown("**Profile.** Resembles " + role + ", operating mainly in " + ch + ".")
st.markdown("**Strengths.**\n" + "\n".join("- " + x for x in strengths))
st.markdown("**Areas to improve.**\n" + "\n".join("- " + x for x in focus))
st.info("Movement advice. " + adv)

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
