/**
 * dashboard.js
 * ------------
 * Interactive ML Visualizations using Chart.js and D3.js.
 */

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("/api/analysis-data");
        const data = await response.json();

        if (data.error) {
            console.error("API Error:", data.error);
            return;
        }

        // --- 1. K-Means Chart (PCA Projection) ---
        window.kmeansChart = initKMeansChart(data.pca_points, data.centroids);

        // --- 2. KNN Chart (Classification Boundaries) ---
        window.knnChart = initKNNChart(data.pca_points);

        // --- 3. Decision Tree Visualization (D3.js) ---
        const treeViz = new TreeViz("#tree-viz", data.tree);
        document.getElementById("grow-tree").addEventListener("click", () => treeViz.grow());
        document.getElementById("reset-tree").addEventListener("click", () => treeViz.reset());

        // --- 4. Random Forest Layered View (D3.js) ---
        const forestViz = new ForestViz("#forest-viz", data.forest);
        document.getElementById("reset-forest").addEventListener("click", () => forestViz.reset());

    } catch (err) {
        console.error("Failed to load dashboard data:", err);
    }
});

/**
 * Global helper for Chart.js Reset Zoom
 */
function resetZoom(chartId) {
    const chart = chartId === 'kmeans-chart' ? window.kmeansChart : window.knnChart;
    if (chart) chart.resetZoom();
}

/**
 * K-Means Scatter Plot
 */
function initKMeansChart(points, centroids) {
    const ctx = document.getElementById("kmeans-chart").getContext("2d");
    const colors = { "Budget": "#00D4AA", "Mid-Range": "#6C63FF", "Premium": "#FF6B6B" };

    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Cars',
                    data: points.map(p => ({ x: p.x, y: p.y, cluster: p.cluster })),
                    backgroundColor: points.map(p => colors[p.cluster] || "#aaa"),
                    pointRadius: 4,
                    hoverRadius: 7
                },
                {
                    label: 'Centroids',
                    data: centroids.map(c => ({ x: c[0], y: c[1] })),
                    backgroundColor: "#000",
                    pointStyle: 'crossRot',
                    pointRadius: 10,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                zoom: {
                    pan: { enabled: true, mode: 'xy' },
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Segment: ${ctx.raw.cluster || 'Centroid'}`
                    }
                }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

/**
 * KNN Scatter Plot
 */
function initKNNChart(points) {
    const ctx = document.getElementById("knn-chart").getContext("2d");
    const colors = { "Budget": "rgba(0, 212, 170, 0.6)", "Mid-Range": "rgba(108, 99, 255, 0.6)", "Premium": "rgba(255, 107, 107, 0.6)" };

    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: ["Budget", "Mid-Range", "Premium"].map(seg => ({
                label: seg,
                data: points.filter(p => p.actual_segment === seg).map(p => ({ x: p.x, y: p.y })),
                backgroundColor: colors[seg],
                pointRadius: 5
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } },
                zoom: {
                    pan: { enabled: true, mode: 'xy' },
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
                }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

/**
 * Decision Tree Visualization Class
 */
class TreeViz {
    constructor(containerId, treeData) {
        this.container = d3.select(containerId);
        this.fullData = treeData;
        this.margin = { top: 40, right: 90, bottom: 50, left: 90 };
        this.width = 1000 - this.margin.left - this.margin.right;
        this.height = 500 - this.margin.top - this.margin.bottom;
        
        this.svgParent = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", "500")
            .attr("viewBox", `0 0 1000 500`);

        this.g = this.svgParent.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        // Add Zoom
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", (event) => {
                this.g.attr("transform", event.transform);
            });
        
        this.svgParent.call(this.zoom);

        this.treemap = d3.tree().size([this.height, this.width]);
        this.root = d3.hierarchy(this.fullData, d => d.children);
        this.root.x0 = this.height / 2;
        this.root.y0 = 0;

        if (this.root.children) {
            this.root._children = this.root.children;
            this.root.children = null;
        }
        
        this.update(this.root);
    }

    reset() {
        this.svgParent.transition().duration(750).call(
            this.zoom.transform,
            d3.zoomIdentity.translate(this.margin.left, this.margin.top)
        );
    }

    grow() {
        const expand = (d) => {
            if (d._children) {
                d.children = d._children;
                d._children = null;
                return true;
            }
            if (d.children) {
                for (let child of d.children) {
                    if (expand(child)) return true;
                }
            }
            return false;
        };
        if (expand(this.root)) this.update(this.root);
    }

    update(source) {
        const treeData = this.treemap(this.root);
        const nodes = treeData.descendants();
        const links = treeData.descendants().slice(1);

        nodes.forEach(d => d.y = d.depth * 180);

        const node = this.g.selectAll("g.node")
            .data(nodes, d => d.id || (d.id = ++this.i || (this.i = 1)));

        const nodeEnter = node.enter().append("g")
            .attr("class", d => `node ${d.data.children ? '' : 'leaf'}`)
            .attr("transform", d => `translate(${source.y0},${source.x0})`);

        nodeEnter.append("circle")
            .attr("r", 1e-6)
            .style("fill", d => d._children ? "var(--accent)" : "#fff");

        nodeEnter.append("text")
            .attr("dy", ".35em")
            .attr("x", d => d.children || d._children ? -13 : 13)
            .attr("text-anchor", d => d.children || d._children ? "end" : "start")
            .text(d => d.data.name + (d.data.threshold ? ` ≤ ${d.data.threshold}` : (d.data.value ? `: ₹${d.data.value.toLocaleString()}` : "")));

        const nodeUpdate = nodeEnter.merge(node);
        nodeUpdate.transition().duration(750)
            .attr("transform", d => `translate(${d.y},${d.x})`);

        nodeUpdate.select("circle")
            .attr("r", 6)
            .style("fill", d => d._children ? "var(--accent)" : "#fff")
            .attr("cursor", "pointer");

        const link = this.g.selectAll("path.link")
            .data(links, d => d.id);

        const linkEnter = link.enter().insert("path", "g")
            .attr("class", "link")
            .attr("d", d => {
                const o = { x: source.x0, y: source.y0 };
                return diagonal(o, o);
            });

        const linkUpdate = linkEnter.merge(link);
        linkUpdate.transition().duration(750)
            .attr("d", d => diagonal(d, d.parent));

        function diagonal(s, d) {
            return `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`;
        }
    }
}

/**
 * Random Forest Layered Visualization Class
 */
class ForestViz {
    constructor(containerId, forestData) {
        this.container = d3.select(containerId);
        this.svgParent = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", "400")
            .attr("viewBox", "0 0 800 400");

        this.g = this.svgParent.append("g");

        // Add Zoom
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on("zoom", (event) => {
                this.g.attr("transform", event.transform);
            });

        this.svgParent.call(this.zoom);

        const colors = ["#6C63FF", "#00D4AA", "#FF6B6B"];
        
        forestData.forEach((tree, i) => {
            const treeGroup = this.g.append("g")
                .attr("transform", `translate(${50 + i * 40}, ${50 + i * 30}) scale(${1 - i * 0.1})`)
                .attr("opacity", 0.8 - i * 0.2);

            const treemap = d3.tree().size([300, 600]);
            const root = d3.hierarchy(tree);
            treemap(root);

            treeGroup.selectAll(".link-rf")
                .data(root.links())
                .enter().append("path")
                .attr("class", "link-rf")
                .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x))
                .attr("fill", "none")
                .attr("stroke", colors[i % colors.length])
                .attr("stroke-width", 1);

            treeGroup.selectAll(".node-rf")
                .data(root.descendants())
                .enter().append("circle")
                .attr("class", "node-rf")
                .attr("cx", d => d.y)
                .attr("cy", d => d.x)
                .attr("r", 2)
                .attr("fill", colors[i % colors.length]);
        });
    }

    reset() {
        this.svgParent.transition().duration(750).call(
            this.zoom.transform,
            d3.zoomIdentity
        );
    }
}

