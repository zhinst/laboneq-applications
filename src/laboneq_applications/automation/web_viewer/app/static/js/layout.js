// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

function setupControls() {
    const btnNode = document.getElementById("btnNodeView");
    const btnLayer = document.getElementById("btnLayerView");
    const btnReset = document.getElementById("btnResetZoom");
    const btnRunAuto = document.getElementById("btnRunAuto");
    const btnResetAuto = document.getElementById("btnResetAuto");

    if (btnNode) btnNode.addEventListener("click", () => setMode("nodes"));
    if (btnLayer) btnLayer.addEventListener("click", () => setMode("layers"));
    if (btnReset) btnReset.addEventListener("click", () => resetZoom());
    if (btnRunAuto) btnRunAuto.addEventListener("click", () => runAutomation());
    if (btnResetAuto)
        btnResetAuto.addEventListener("click", () => resetAutomation());

    updateViewToggleUI();
}

function updateViewToggleUI() {
    const btnNode = document.getElementById("btnNodeView");
    const btnLayer = document.getElementById("btnLayerView");

    if (!btnNode || !btnLayer) return;

    btnNode.classList.toggle("active", currentMode === "nodes");
    btnLayer.classList.toggle("active", currentMode === "layers");
}

async function resetAutomation() {
    const resp = await fetch("/reset", { method: "POST" });
    const data = await resp.json().catch(() => ({}));
    if (resp.status == 202) {
        refreshData(true);
        d3.select("#status")
            .html("<strong>Automation reset</strong>!")
            .style("display", "block");
    }
}

async function runAutomation() {
    d3.select("#status")
        .html("<strong>Automation:</strong> run started!")
        .style("display", "block");

    const resp = await fetch("/run", { method: "POST" });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        d3.select("#status")
            .html("<strong>Automation:</strong> run finished!")
            .style("display", "block");
        return;
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

    g.selectAll(".link").style("stroke", (d) =>
        d3.color(colorPalette.gray).darker(1.5),
    );

    g.selectAll(".layer-label").remove();

    const match = g.selectAll("g.node").filter((d) => d.layer === layerKey);
    if (!match.empty()) {
        g.append("text")
            .attr("class", "layer-label")
            .attr("transform", match.attr("transform"))
            .attr("dx", -30)
            .attr("dy", -50)
            .attr("text-anchor", "left")
            .style("fill", "black")
            .style("font-size", "40px")
            .style("font-weight", "bold")
            .style("pointer-events", "none")
            .text(layerKey);
    }
}

function clearHighlight() {
    g.selectAll("g.node circle")
        .attr("fill", (d) => statusColorMap[d.status] || colorPalette.zi_blue)
        .style("filter", null);
    g.selectAll(".link").style("stroke", null);
    g.selectAll(".layer-label").remove();
}
