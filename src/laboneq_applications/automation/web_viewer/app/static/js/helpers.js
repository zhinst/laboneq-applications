function getContainerSize() {
    const el = document.getElementById("graph");
    return { width: el.clientWidth, height: el.clientHeight };
}

const header = document.getElementById("header");
const ro = new ResizeObserver(() => {
    document.documentElement.style.setProperty(
        "--header-h",
        header.offsetHeight + "px",
    );
});
ro.observe(header);

function hideCanvas(canvas, duration = 0) {
    g.select(canvas)
        .transition()
        .duration(duration)
        .style("opacity", 0)
        .transition()
        .duration(duration)
        .style("visibility", "hidden");
}

function showCanvas(canvas, duration = 0) {
    const sel = g.select(canvas);

    sel.interrupt()
        .style("visibility", "visible")
        .style("pointer-events", "auto")
        .transition()
        .duration(duration)
        .style("opacity", 1);
}
