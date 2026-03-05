// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

let currentVersion = null;
let svg, g, zoom;
let lastSelectedNode = null;
let currentMode = "nodes";
let cachedGraphData = null;
let isTransitioning = false;

const TRANSITION_DURATION = 750;

const statusColorMap = {
    root: "#009EE0",
    running: "#EFBF04",
    passed: "#68E000",
    failed: "#E05100",
    ready: "#E09700",
    mixed: "#615130",
    deactivated: "#7f7f7f",
};

async function fetchGraphData() {
    try {
        const response = await fetch("/graph");
        return response.ok ? await response.json() : null;
    } catch (e) {
        console.error("Error fetching graph data:", e);
        return null;
    }
}

function setupSVG() {
    const container = document.getElementById("graph");

    svg = d3
        .select("#graph")
        .append("svg")
        .attr("width", container.clientWidth)
        .attr("height", container.clientHeight)
        .on("click", function (event) {
            if (event.target === this) {
                d3.select("#status").style("display", "none");
                if (lastSelectedNode) {
                    lastSelectedNode.classed("selected", false);
                    lastSelectedNode = null;
                    clearHighlight();
                }
            }
        });

    g = svg.append("g");

    zoom = d3
        .zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => g.attr("transform", event.transform));

    svg.call(zoom).call(zoom.transform, d3.zoomIdentity);
}

function computeScales(items, width, height, pad = 0.2) {
    const xs = items.map((n) => n.x);
    const ys = items.map((n) => n.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    let xPad = pad * width;
    let yPad = pad * height;

    if (Math.abs(maxX - minX) < 0.5) {
        xPad = (width - 0.5 * width) / 2;
    }
    if (Math.abs(maxY - minY) < 0.7 * height) {
        yPad = (height - 0.7 * height) / 2;
    }

    return {
        xScale: d3
            .scaleLinear()
            .domain([minX, maxX])
            .range([xPad, width - xPad]),
        yScale: d3
            .scaleLinear()
            .domain([minY, maxY])
            .range([height - yPad, yPad]),
    };
}

function itemPositions(items, xScale, yScale) {
    const pos = {};
    items.forEach((n) => {
        pos[n.key] = { x: xScale(n.x), y: yScale(n.y) };
    });
    return pos;
}

function buildLayerMap(layers) {
    return new Map(layers.map((layer, i) => [layer.key, i, layer.status]));
}

function getContainerSize() {
    const container = document.getElementById("graph");
    return { width: container.clientWidth, height: container.clientHeight };
}

function getItemsAndLinks(graphData, mode) {
    const isLayers = mode === "layers";
    return {
        items: isLayers ? graphData.layers : graphData.nodes,
        links: isLayers ? graphData.layer_links : graphData.node_links,
        isLayers,
    };
}

/**
 * Compute positions for a given mode.
 */
function computeLayout(graphData, mode) {
    const { width, height } = getContainerSize();

    const nodes = graphData.nodes || [];
    const layers = graphData.layers || [];

    if (!nodes.length || !layers.length) {
        const { items } = getItemsAndLinks(graphData, mode);
        if (!items?.length)
            return {
                items: [],
                links: [],
                pos: {},
                isLayers: mode === "layers",
            };

        const { xScale, yScale } = computeScales(items, width, height);
        const pos = itemPositions(items, xScale, yScale);
        const { links, isLayers } = getItemsAndLinks(graphData, mode);
        return { items, links, pos, isLayers };
    }

    // Compute both node and layer positions so we can animate between them
    const nodeScale = computeScales(nodes, width, height);
    const layerScale = computeScales(layers, width, height);

    const nodePos = itemPositions(nodes, nodeScale.xScale, nodeScale.yScale);
    const layerPos = itemPositions(
        layers,
        layerScale.xScale,
        layerScale.yScale,
    );

    const toLayers = mode === "layers";

    if (toLayers) {
        const { items, links, isLayers } = getItemsAndLinks(graphData, mode);
        return { items, links, pos: layerPos, isLayers };
    }

    const { items, links, isLayers } = getItemsAndLinks(graphData, mode);
    return { items, links, pos: nodePos, isLayers };
}

function computeTransitionNodePositions(graphData) {
    const { width, height } = getContainerSize();

    const nodes = graphData.nodes || [];
    const layers = graphData.layers || [];
    if (!nodes.length || !layers.length) return null;

    const nodeScale = computeScales(nodes, width, height);
    const layerScale = computeScales(layers, width, height);

    const nodePos = itemPositions(nodes, nodeScale.xScale, nodeScale.yScale);
    const layerPos = itemPositions(
        layers,
        layerScale.xScale,
        layerScale.yScale,
    );

    const nodeLayerPos = {};
    nodes.forEach((n) => {
        nodeLayerPos[n.key] = layerPos[n.layer] || nodePos[n.key];
    });

    return { nodePos, nodeLayerPos };
}

function clearScene() {
    g.selectAll("*").remove();
}

function renderLinks({
    links,
    pos,
    keyFn = (d) => `${d[0]}->${d[1]}`,
    initialPos = null,
    animate = false,
}) {
    const startPos = initialPos || pos;

    const sel = g
        .selectAll("line.link")
        .data(links || [], keyFn)
        .join(
            (enter) =>
                enter
                    .append("line")
                    .attr("class", "link")
                    .attr("x1", (d) => startPos[d[0]]?.x ?? 0)
                    .attr("y1", (d) => startPos[d[0]]?.y ?? 0)
                    .attr("x2", (d) => startPos[d[1]]?.x ?? 0)
                    .attr("y2", (d) => startPos[d[1]]?.y ?? 0),
            (update) => update,
            (exit) => exit.remove(),
        );

    if (animate) {
        sel.transition()
            .duration(TRANSITION_DURATION)
            .attr("x1", (d) => pos[d[0]]?.x ?? 0)
            .attr("y1", (d) => pos[d[0]]?.y ?? 0)
            .attr("x2", (d) => pos[d[1]]?.x ?? 0)
            .attr("y2", (d) => pos[d[1]]?.y ?? 0);
    } else {
        sel.attr("x1", (d) => pos[d[0]]?.x ?? 0)
            .attr("y1", (d) => pos[d[0]]?.y ?? 0)
            .attr("x2", (d) => pos[d[1]]?.x ?? 0)
            .attr("y2", (d) => pos[d[1]]?.y ?? 0);
    }
}

function renderNodes({
    items,
    pos,
    layerMap,
    isLayers,
    initialPos = null,
    animate = false,
    toMode = "nodes",
}) {
    const startPos = initialPos || pos;
    const toLayers = toMode === "layers";

    const nodeSel = g
        .selectAll("g.node")
        .data(items || [], (d) => d.key)
        .join(
            (enter) => {
                const ng = enter
                    .append("g")
                    .attr("class", "node")
                    .attr(
                        "transform",
                        (d) =>
                            `translate(${startPos[d.key]?.x ?? 0},${startPos[d.key]?.y ?? 0})`,
                    );
                ng.append("circle")
                    .attr("class", "node-circle")
                    .attr("r", (d) =>
                        Array.isArray(d.quantum_elements) && !toLayers
                            ? d.quantum_elements.length * 20
                            : 30,
                    )
                    .attr("fill", (d) => statusColorMap[d.status] || "#cccccc")
                    .attr("stroke", "#999")
                    .attr("stroke-width", 2)
                    .on("click", (event, d) => {
                        if (lastSelectedNode)
                            lastSelectedNode.classed("selected", false);
                        const cur = d3.select(event.currentTarget.parentNode);
                        cur.classed("selected", true);
                        lastSelectedNode = cur;

                        highlightLayer(d.layer);

                        const qe = Array.isArray(d.quantum_elements)
                            ? d.quantum_elements.join(", ")
                            : d.quantum_elements;

                        d3.select("#status")
                            .html(
                                `<strong>${isLayers ? "Layer" : "Node"}:</strong> ${d.key}<br>` +
                                    `<strong>Status:</strong> ${d.status}<br>` +
                                    `<strong>Layer:</strong> ${d.layer}<br>` +
                                    `<strong>Elements:</strong> ${qe}<br>` +
                                    `<strong>Timestamp:</strong> ${d.timestamp || "N/A"}<br>` +
                                    `<strong>Fail count:</strong> ${d.fail_count}<br>` +
                                    `<strong>Depends on:</strong> ${d.depends_on}`,
                            )
                            .style("display", "block");
                    });

                ng.on("mouseenter", function () {
                    d3.select(this)
                        .select("circle")
                        .style("stroke", "#000")
                        .style("stroke-width", 3);
                    d3.select(this).raise();
                }).on("mouseleave", function () {
                    d3.select(this)
                        .select("circle")
                        .style("stroke", null)
                        .style("stroke-width", null);
                    g.selectAll("g.node").sort(
                        (a, b) => items.indexOf(a) - items.indexOf(b),
                    );
                });

                ng.append("text")
                    .attr("class", "top-label")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "top")
                    .attr("fill", "black")
                    .style("font-size", "25px")
                    .style("pointer-events", "none")
                    .text((d) => layerMap.get(d.layer));

                // Qubit key label exists only for nodes-mode view of nodes,
                // but for transitions we keep it and fade it in/out.
                ng.append("text")
                    .attr("class", "bottom-label")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "central")
                    .attr("fill", "black")
                    .attr("y", 15)
                    .style("font-size", "18px")
                    .style("pointer-events", "none")
                    .style("opacity", toLayers ? 1 : 0)
                    .text((d) =>
                        Array.isArray(d.quantum_elements)
                            ? d.quantum_elements.join(", ")
                            : d.quantum_elements,
                    );

                return ng;
            },
            (update) => update,
            (exit) => exit.remove(),
        );

    // For settled modes: hide qubit key label when in layers, show when in nodes
    if (!animate) {
        nodeSel.select(".bottom-label").style("opacity", toLayers ? 0 : 1);
    }

    if (animate) {
        // Move nodes
        nodeSel
            .transition()
            .duration(TRANSITION_DURATION)
            .attr(
                "transform",
                (d) => `translate(${pos[d.key]?.x ?? 0},${pos[d.key]?.y ?? 0})`,
            );

        // Fade out qubit label
        nodeSel
            .select(".bottom-label")
            .transition()
            .duration(TRANSITION_DURATION)
            .style("opacity", toLayers ? 0 : 1);

        // Resize circle
        nodeSel
            .select(".node-circle")
            .interrupt()
            .transition()
            .duration(TRANSITION_DURATION)
            .attr("r", (d) =>
                Array.isArray(d.quantum_elements) && !toLayers
                    ? d.quantum_elements.length * 20
                    : 30,
            );
    }
}

/**
 * Single entry-point renderer.
 *
 * - Settled: animateFromMode = null
 * - Transition: animateFromMode = "nodes" or "layers"
 */
function renderGraph(graphData, mode, { animateFromMode = null } = {}) {
    const { items, links, pos, isLayers } = computeLayout(graphData, mode);
    if (!items?.length) return;

    const layerMap = buildLayerMap(graphData.layers || []);
    renderLayerLegend(graphData.layers || [], layerMap);

    clearScene();

    const animate = animateFromMode !== null;

    // Settled:
    // - nodes mode: render nodes + node_links
    // - layers mode: render layers + layer_links
    if (!animate) {
        renderLinks({ links, pos });
        renderNodes({
            items,
            pos,
            layerMap,
            isLayers,
            animate: false,
            toMode: mode,
        });

        if (mode === "layers") {
            g.selectAll("g.node .bottom-label").remove();
        }
        return;
    }

    // Transition nodes <--> layers:
    const trans = computeTransitionNodePositions(graphData);
    if (!trans) {
        renderGraph(graphData, mode, { animateFromMode: null });
        return;
    }

    const toLayers = mode === "layers";
    const startPos = toLayers ? trans.nodePos : trans.nodeLayerPos;
    const endPos = toLayers ? trans.nodeLayerPos : trans.nodePos;

    renderLinks({
        links: graphData.node_links || [],
        pos: endPos,
        initialPos: startPos,
        animate: true,
    });
    renderNodes({
        items: graphData.nodes || [],
        pos: endPos,
        initialPos: startPos,
        layerMap,
        isLayers: false,
        animate: true,
        toMode: mode,
    });

    isTransitioning = true;
    // End of transition: swtich to settled final view
    window.setTimeout(() => {
        renderGraph(graphData, mode, { animateFromMode: null });
        isTransitioning = false;
    }, TRANSITION_DURATION);
}

function resetZoom() {
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}

function setMode(mode) {
    if (currentMode === mode || isTransitioning) return;

    const prevMode = currentMode;
    currentMode = mode;
    updateViewToggleUI();

    if (cachedGraphData) {
        renderGraph(cachedGraphData, mode, { animateFromMode: prevMode });
    } else {
        refreshData({ force: true });
    }
}

async function refreshData({ force = false } = {}) {
    if (isTransitioning) return;

    const graphData = await fetchGraphData();
    if (!graphData) return;

    if (!force && currentVersion === graphData.version) return;

    currentVersion = graphData.version;
    cachedGraphData = graphData;
    renderGraph(graphData, currentMode, { animateFromMode: null });
}

async function init() {
    setupSVG();
    setupControls();
    renderStatusLegend();
    await refreshData({ force: true });

    setInterval(refreshData, 200);

    window.addEventListener("resize", () => {
        const { width, height } = getContainerSize();
        svg.attr("width", width).attr("height", height);
        if (!isTransitioning && cachedGraphData) {
            renderGraph(cachedGraphData, currentMode, {
                animateFromMode: null,
            });
        }
    });
}

init();
