// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

let toastTimer = null;

function setupControls() {
    const toggle = document.getElementById("viewToggle");
    const btnReset = document.getElementById("btnResetZoom");
    const btnRunAuto = document.getElementById("btnRunAuto");
    const btnResetAuto = document.getElementById("btnResetAuto");

    toggle?.addEventListener("click", () => {
        if (toggle.dataset.locked) return;
        const next = toggle.dataset.active === "layers" ? "nodes" : "layers";
        switchMode(toggle, next);
    });

    if (btnReset) btnReset.addEventListener("click", resetZoom);
    if (btnRunAuto) btnRunAuto.addEventListener("click", runAutomation);
    if (btnResetAuto) btnResetAuto.addEventListener("click", resetAutomation);
}

function switchMode(toggle, mode) {
    toggle.dataset.active = mode;
    toggle.dataset.locked = true;

    clearHighlight();
    setMode(mode);

    setTimeout(() => {
        delete toggle.dataset.locked;
    }, TRANSITION_DURATION);
}

function showToast(html, { persistent = false, duration = 2500 } = {}) {
    const el = document.getElementById("automation-status");

    clearTimeout(toastTimer);
    el.classList.remove("fading");
    el.style.display = "block";
    el.innerHTML = html;

    if (persistent) return;

    toastTimer = setTimeout(() => {
        el.classList.add("fading");
        toastTimer = setTimeout(() => {
            el.style.display = "none";
            el.classList.remove("fading");
        }, 500); // matches transition duration
    }, duration);
}

async function resetAutomation() {
    if (!confirm("Are you sure you want to reset the automation?")) return;

    const resp = await fetch("/reset", { method: "POST" });
    const data = await resp.json().catch(() => ({}));
    if (resp.status == 202) {
        refreshData(true);
        showToast("<strong>Automation reset</strong>!");
    }
    clearHighlight();
}

async function runAutomation() {
    clearHighlight();

    showToast("<strong>Automation:</strong> running...", { persistent: true });

    const resp = await fetch("/run", { method: "POST" });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("<strong>Automation:</strong> run finished!");
    }
}

function highlightLayer(layerKey) {
    g.selectAll("g.node circle")
        .attr("fill", (d) => {
            const color = statusColorMap[d.status] || colorPalette.zi_blue;
            return d.layer === layerKey ? color : d3.color(color).darker(1.5);
        })
        .style("filter", (d) => {
            if (d.layer !== layerKey) return null;
            const color = statusColorMap[d.status] || colorPalette.zi_blue;
            return `drop-shadow(0 0 8px ${color})`;
        });

    g.selectAll(".layer-label").remove();
    const match = g.selectAll("g.node").filter((d) => d.layer === layerKey);
    if (!match.empty()) {
        let minX = Infinity;
        let leftmostNode = null;

        match.each(function () {
            const t = d3.select(this).attr("transform");
            const m = t && t.match(/translate\(([^,)]+)/);
            if (m) {
                const x = parseFloat(m[1]);
                if (x < minX) {
                    minX = x;
                    leftmostNode = this;
                }
            }
        });

        const nodeTransform = d3.select(leftmostNode).attr("transform");
        const nodeBBox = leftmostNode.getBBox();
        const gap = 16;
        const px = 6;

        const label = g
            .append("g")
            .attr("class", "layer-label")
            .attr("transform", nodeTransform);

        const text = label
            .append("text")
            .attr("x", nodeBBox.x - gap)
            .attr("dy", "0.35em")
            .attr("text-anchor", "end")
            .attr("font-size", nodeBBox.height * 0.7)
            .text(layerKey);

        const textBBox = text.node().getBBox();

        label
            .insert("rect", "text")
            .attr("x", textBBox.x - px)
            .attr("y", nodeBBox.y)
            .attr("width", textBBox.width + px * 2)
            .attr("height", nodeBBox.height)
            .attr("rx", textBBox.height / 4)
            .attr("ry", textBBox.height / 4);
    }
}

function clearHighlight() {
    g.selectAll("g.node circle")
        .attr("fill", (d) => statusColorMap[d.status] || colorPalette.zi_blue)
        .style("filter", null);
    g.selectAll(".link").style("stroke", null);
    g.selectAll(".layer-label").remove();
}
