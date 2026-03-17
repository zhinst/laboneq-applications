function renderStatusLegend() {
    const container = d3.select("#status-legend-items");
    container.selectAll("*").remove();

    const entries = Object.entries(statusColorMap);

    const row = container
        .selectAll(".legend-row")
        .data(entries, (d) => d[0])
        .enter()
        .append("div")
        .attr("class", "legend-row");

    const left = row.append("div").attr("class", "legend-left");

    left.append("div")
        .attr("class", "legend-dot")
        .style("background-color", (d) => d[1]);

    left.append("span").text(([status]) =>
        status === "__root__"
            ? "Root"
            : status.charAt(0).toUpperCase() + status.slice(1),
    );
}

function renderLayerLegend(layers, layerMap) {
    const container = d3.select("#layer-legend-items");
    container.selectAll("*").remove();

    const row = container
        .selectAll(".legend-row")
        .data(layers, (d) => d.key)
        .enter()
        .append("div")
        .attr("class", "legend-row");

    const left = row.append("div").attr("class", "legend-left");

    left.append("div")
        .attr("class", "legend-badge")
        .style(
            "background-color",
            (d) => statusColorMap[d.status] || colorPalette.zi_blue,
        )
        .text((d) => layerMap.get(d.key));

    left.append("span").text((d) =>
        d.key === "__root__"
            ? "Root"
            : d.key.charAt(0).toUpperCase() + d.key.slice(1),
    );
}

function updateNodeInfoLegend(d, qe, isLayers) {
    d3.select("#status-title").text(isLayers ? "Layer info" : "Node info");

    const rows = [
        { label: "Key", value: d.key },
        { label: "Status", value: d.status },
        ...(isLayers ? [] : [{ label: "Layer", value: d.layer }]),
        ...(isLayers ? [{ label: "Sequential", value: d.sequential }] : []),
        { label: "Elements", value: qe },
        { label: "Timestamp", value: d.timestamp || "N/A" },
        { label: "Fail count", value: d.fail_count },
        { label: "Pass count", value: d.pass_count },
        { label: "Depends on", value: d.depends_on.join(", ") || "root" },
    ];

    const sel = d3
        .select("#node-info-rows")
        .selectAll(".legend-row")
        .data(rows, (r) => r.label);

    sel.exit().remove();

    const enter = sel.enter().append("div").attr("class", "legend-row");

    enter.append("div").attr("class", "legend-left");
    enter.append("div").attr("class", "legend-right");

    const merged = enter.merge(sel);

    merged.select(".legend-left").text((r) => `${r.label}:`);

    merged.select(".legend-right").text((r) => r.value ?? "");

    d3.select("#node-info").style("display", "block");
}
