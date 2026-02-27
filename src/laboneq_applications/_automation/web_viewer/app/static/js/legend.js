function renderStatusLegend() {
    const legendContainer = d3.select("#status-legend-items");
    legendContainer.selectAll("*").remove();

    const entries = Object.entries(statusColorMap);

    entries.forEach(([status, color]) => {
        const label =
            status === "__root__"
                ? "Root"
                : status.charAt(0).toUpperCase() + status.slice(1);

        const item = legendContainer.append("div").attr("class", "legend-item");

        item.append("div")
            .attr("class", "legend-color")
            .style("background-color", color);

        item.append("span").text(label);
    });
}

function renderLayerLegend(layers, layerMap) {
    const legendContainer = d3.select("#layer-legend-items");
    legendContainer.selectAll("*").remove();

    layers.forEach((layer) => {
        const number = layerMap.get(layer.key);
        const label =
            layer.key === "__root__"
                ? "Root"
                : layer.key.charAt(0).toUpperCase() + layer.key.slice(1);

        console.log(layer);
        console.log(statusColorMap[layer.status]);
        const color = statusColorMap[layer.status];
        const item = legendContainer.append("div").attr("class", "legend-item");

        item.append("div")
            .attr("class", "legend-number")
            .style("background-color", color)
            .text(number);

        item.append("span").text(label);
    });
}
