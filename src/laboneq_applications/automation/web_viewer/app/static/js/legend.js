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
        status === "root"
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
        d.key === "root"
            ? "root"
            : d.key,
    );
}

function openImageModal(src) {
    document.getElementById("image-modal-img").src = src;
    document.getElementById("image-modal").style.display = "flex";
    document.getElementById("graph").classList.add("modal-open");
}

function closeImageModal() {
    document.getElementById("image-modal").style.display = "none";
    document.getElementById("image-modal-img").src = "";
    document.getElementById("graph").classList.remove("modal-open");
}

document.getElementById("image-modal").addEventListener("click", closeImageModal);

function updateNodeInfoLegend(d, qe, isLayers) {
    d3.select("#node-results-section").remove();
    d3.select("#status-title").text(isLayers ? "Layer info" : "Node info");

    const rows = [
        ...(isLayers ? [{ label: "Key", value: d.key }] : [{ label: "ID", value: d.key }]),
        { label: "Status", value: d.status },
        ...(isLayers ? [] : [{ label: "Layer", value: d.layer }]),
        ...(isLayers ? [{ label: "Sequential", value: d.key === "root" ? "N/A" : d.sequential }] : []),
        { label: "Elements", value: qe },
        { label: "Timestamp", value: d.timestamp || "N/A" },
        { label: "Fail count", value: (d.key === "root" || d.key === "root_root") ? "N/A" : d.fail_count },
        { label: "Pass count", value: (d.key === "root" || d.key === "root_root") ? "N/A" : d.pass_count },
        ...(isLayers ? [{ label: "Depends on", value: d.key === "root" ? "N/A" : d.depends_on.join(", ") || "root" }] :
            [{ label: "Depends on", value: (d.key === "root_root") ? "N/A" : d.depends_on.join(", ") || "root_root" }]),
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

    if (!isLayers && (d.status === "passed" || d.status === "failed") && cachedGraphData?.has_log_path) {
        const qeKey = Array.isArray(d.quantum_elements)
            ? d.quantum_elements.join("-")
            : String(d.quantum_elements);
        const resultsSection = d3
            .select("#node-info")
            .append("div")
            .attr("id", "node-results-section");
        resultsSection.append("h3").text("Node results");
        const url = `/node-image?layer=${encodeURIComponent(d.layer)}&qe=${encodeURIComponent(qeKey)}`;
        fetch(url)
            .then((resp) => {
                if (resp.status === 204) {
                    resultsSection.append("span").text("File not found");
                    return null;
                }
                if (!resp.ok) {
                    d3.select("#node-results-section").remove();
                    return null;
                }
                return resp.blob();
            })
            .then((blob) => {
                if (!blob) return;
                const objectUrl = URL.createObjectURL(blob);
                resultsSection
                    .append("img")
                    .attr("class", "node-result-thumbnail")
                    .attr("src", objectUrl)
                    .on("click", () => openImageModal(objectUrl));
            })
            .catch(() => d3.select("#node-results-section").remove());
    }
}
