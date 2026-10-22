#!/usr/bin/env python3
"""Render measured paths as an SVG comparison and an offline HTML path player."""
import argparse
import html
import json
import pathlib

COLORS = ["#2866db", "#1b8870", "#ad4b96", "#cf6a26", "#7859c1", "#c54855"]


def cell_color(c):
    return "#243247" if c == "#" else "#ffe8c7" if c.isdigit() else "#f0f3f8"


def svg_comparison(grid, rows):
    selected = ["bfs", "dijkstra", "astar", "weighted-astar", "greedy", "dfs"]
    rows = [next((r for r in rows if r["algorithm"] == a), None) for a in selected]
    rows = [r for r in rows if r is not None]
    w, h, size, card_w, card_h = len(grid[0]), len(grid), 25, 288, 340
    height = 106 + ((len(rows) + 2) // 3) * card_h
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="936" height="%d" viewBox="0 0 936 %d">' % (height, height),
             '<rect width="936" height="%d" fill="#eef2f8"/>' % height,
             '<text x="30" y="38" font-family="Arial,sans-serif" font-size="25" font-weight="bold" fill="#18253a">CudaSearchLab / One map, different decisions</text>',
             '<text x="30" y="66" font-family="Arial,sans-serif" font-size="13" fill="#52627a">Actual CPU results. Orange cells cost 9 to enter; other walkable cells cost 1.</text>']
    for i, row in enumerate(rows):
        x, y, color = 24 + (i % 3) * 304, 90 + (i // 3) * card_h, COLORS[i % len(COLORS)]
        parts += ['<rect x="%d" y="%d" width="%d" height="%d" rx="14" fill="white"/>' % (x, y, card_w, card_h - 16),
                  '<text x="%d" y="%d" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="%s">%s</text>' %
                  (x + 18, y + 29, color, html.escape(row["algorithm"])),
                  '<text x="%d" y="%d" font-family="Arial,sans-serif" font-size="12" fill="#52627a">Cost %s / %d hops / %d expansions</text>' %
                  (x + 18, y + 51, row["cost"], max(0, len(row["path"]) - 1), row["expansions"])]
        gx, gy = x + (card_w - w * size) / 2, y + 72
        for yy, line in enumerate(grid):
            for xx, c in enumerate(line):
                parts.append('<rect x="%g" y="%g" width="%d" height="%d" rx="3" fill="%s" stroke="white"/>' %
                             (gx + xx * size, gy + yy * size, size, size, cell_color(c)))
                if c.isdigit():
                    parts.append('<text x="%g" y="%g" text-anchor="middle" font-family="Arial" font-size="11" fill="#9a6032">%s</text>' %
                                 (gx + (xx + .5) * size, gy + (yy + .66) * size, c))
        points = " ".join("%g,%g" % (gx + (v % w + .5) * size, gy + (v // w + .5) * size) for v in row["path"])
        if points:
            parts.append('<polyline points="%s" fill="none" stroke="white" stroke-width="7" stroke-linejoin="round"/>' % points)
            parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="3.5" stroke-linejoin="round"/>' % (points, color))
            for j, v in enumerate([row["path"][0], row["path"][-1]]):
                cx, cy = gx + (v % w + .5) * size, gy + (v // w + .5) * size
                parts.append('<circle cx="%g" cy="%g" r="9" fill="%s"/>' % (cx, cy, color))
                parts.append('<text x="%g" y="%g" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">%s</text>' %
                             (cx, cy + 3.5, "S" if j == 0 else "G"))
        parts.append('<text x="%d" y="%d" font-family="Arial,sans-serif" font-size="12" fill="#52627a">Status: %s</text>' %
                     (x + 18, y + card_h - 35, html.escape(row["status"])))
    return "\n".join(parts + ["</svg>"])


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CudaSearchLab / Path explorer</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf1f7;color:#19263b;font:15px system-ui,sans-serif}
header{background:#18263d;color:white;padding:26px 6vw}header small{color:#a7bad8;letter-spacing:.15em}
h1{font-size:30px;margin:8px 0}header p{color:#c1cfe2;margin-bottom:0}
main{max-width:1080px;margin:30px auto;padding:0 20px;display:grid;grid-template-columns:280px 1fr;gap:24px}
aside,article{background:white;border-radius:18px;padding:24px;box-shadow:0 8px 32px #20345008}
label{display:block;color:#58667b;font-size:13px;margin-bottom:8px}
select,button{width:100%;padding:12px;border:1px solid #d7dfeb;border-radius:8px;background:#f7f9fc;color:inherit;font:inherit}
button{background:#2866db;color:white;border:0;margin:16px 0;cursor:pointer}input{width:100%;accent-color:#2866db}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0}.stats div{background:#f1f4f9;border-radius:10px;padding:12px}
.stats strong{display:block;font-size:24px;margin-top:3px}.stats span{font-size:12px;color:#66768c}
#board{display:block;width:100%;max-height:600px}footer{font-size:12px;color:#66768c;line-height:1.7}
article p{color:#66768c;font-size:13px}#status{display:inline-block;padding:5px 10px;border-radius:8px;background:#e2f2e9;color:#287758}
@media(max-width:700px){main{grid-template-columns:1fr}h1{font-size:25px}}
</style>
<header><small>CUDA C++ / CLASSICAL SEARCH</small><h1>One graph. Fifteen ways to search.</h1>
<p>Inspect the final paths returned by the algorithms.</p></header>
<main><aside><label for="algorithm">Algorithm / query</label><select id="algorithm"></select>
<div class="stats"><div><span>Path cost</span><strong id="cost"></strong></div><div><span>Hops</span><strong id="hops"></strong></div>
<div><span>Expansions</span><strong id="expanded"></strong></div><div><span>Backend</span><strong id="backend" style="font-size:18px"></strong></div></div>
<span id="status"></span><button id="play">Play path</button><label for="step">Path prefix</label><input id="step" type="range" min="0" value="0">
<footer>This player reveals the returned route. It does not show expansion order.
Orange cells are terrain costs. S and G mark the query endpoints.
Timing includes allocation, transfers, host orchestration, and reconstruction; see the benchmark documentation.</footer></aside>
<article><svg id="board" role="img" aria-label="Graph path on the input grid"></svg><p id="caption"></p></article></main>
<script>
"use strict";
const data=@@DATA@@;
const select=document.getElementById("algorithm"), step=document.getElementById("step");
let timer=null;
data.rows.forEach(function(r,i){const option=document.createElement("option");option.value=i;
option.textContent=r.algorithm+" / q"+r.query;select.appendChild(option);});
function element(tag,attrs,text){const el=document.createElementNS("http://www.w3.org/2000/svg",tag);
Object.entries(attrs).forEach(function(kv){el.setAttribute(kv[0],kv[1]);});if(text!==undefined)el.textContent=text;return el;}
function draw(reset){const row=data.rows[Number(select.value)], board=document.getElementById("board"),w=data.grid[0].length,h=data.grid.length;
if(reset){step.max=row.path.length;step.value=row.path.length;}
const count=Number(step.value);board.replaceChildren();board.setAttribute("viewBox","0 0 "+w*40+" "+h*40);
data.grid.forEach(function(line,y){Array.from(line).forEach(function(c,x){
board.appendChild(element("rect",{x:x*40+1,y:y*40+1,width:38,height:38,rx:5,fill:c==="#"?"#243247":/[1-9]/.test(c)?"#ffe8c7":"#edf1f7"}));
if(/[1-9]/.test(c))board.appendChild(element("text",{x:x*40+20,y:y*40+25,"text-anchor":"middle",fill:"#9a6032","font-size":15},c));});});
const points=row.path.slice(0,count).map(function(v){return (v%w*40+20)+","+(Math.floor(v/w)*40+20);}).join(" ");
if(points)board.appendChild(element("polyline",{points:points,fill:"none",stroke:"#2866db","stroke-width":6,"stroke-linecap":"round","stroke-linejoin":"round"}));
[row.source,row.goal].forEach(function(v,i){const x=v%w*40+20,y=Math.floor(v/w)*40+20;
board.appendChild(element("circle",{cx:x,cy:y,r:12,fill:i?"#d2614f":"#1b8870"}));
board.appendChild(element("text",{x:x,y:y+4,"text-anchor":"middle",fill:"white","font-size":12,"font-weight":700},i?"G":"S"));});
document.getElementById("cost").textContent=row.cost===null?"—":row.cost;
document.getElementById("hops").textContent=row.path.length?row.path.length-1:"—";
document.getElementById("expanded").textContent=row.expansions;document.getElementById("backend").textContent=row.backend;
document.getElementById("status").textContent=row.status;
document.getElementById("caption").textContent="Showing "+count+" of "+row.path.length+" path vertices. Objective: "+row.objective+".";
}
function stop(){if(timer!==null){clearInterval(timer);timer=null;}document.getElementById("play").textContent="Play path";}
select.addEventListener("change",function(){stop();draw(true);});step.addEventListener("input",function(){stop();draw(false);});
document.getElementById("play").addEventListener("click",function(){if(timer!==null){stop();return;}step.value=0;draw(false);
this.textContent="Pause";timer=setInterval(function(){step.value=Number(step.value)+1;draw(false);if(Number(step.value)>=Number(step.max))stop();},130);});
draw(true);
</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=pathlib.Path, required=True)
    p.add_argument("--results", type=pathlib.Path, required=True)
    p.add_argument("--svg", type=pathlib.Path, default=pathlib.Path("docs/path_comparison.svg"))
    p.add_argument("--html", type=pathlib.Path, default=pathlib.Path("docs/path_explorer.html"))
    args = p.parse_args()
    grid = args.grid.read_text().splitlines()
    if not grid or not grid[0] or any(len(line) != len(grid[0]) for line in grid):
        p.error("grid must be nonempty and rectangular")
    rows = json.loads(args.results.read_text())
    if not rows:
        p.error("result list is empty")
    n = len(grid) * len(grid[0])
    for row in rows:
        ids = row["path"] + [row["source"], row["goal"]]
        if any(not isinstance(v, int) or v < 0 or v >= n for v in ids):
            p.error("path/query vertex outside grid")
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg_comparison(grid, rows))
    payload = json.dumps({"grid": grid, "rows": rows}).replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    args.html.write_text(PAGE.replace("@@DATA@@", payload))
    print(args.svg)
    print(args.html)


if __name__ == "__main__":
    main()
