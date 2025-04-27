// D3 v7 uses d3.select, etc. directly
// Global variables
let simulation = null;
let graphData = { nodes: [], links: [] }; // Store current graph data
let currentNode = null; // Track the currently displayed page title
let currentMode = 'extract_links'; // 'extract_links' or 'add_node'
let autoRefreshIntervalId = null;
const AUTO_REFRESH_INTERVAL_MS = 10000; // 10 seconds

// D3 selections
const graphContainer = d3.select("#graph-container");
const graphLoadingSpinner = d3.select("#graph-loading");
const contentTitle = d3.select("#content-title");
const contentBody = d3.select("#content-body");
const analysisResultsContainer = d3.select("#analysis-results-container");
const searchInput = d3.select("#search-input");
const searchBtn = d3.select("#search-btn");
const suggestionsList = d3.select("#suggestions");
const randomArticleBtn = d3.select("#random-article-btn");
const modeToggle = d3.select("#mode-toggle");
const modeLabel = d3.select("#mode-label");
const modeDescription = d3.select("#mode-description");
const clearGraphBtn = d3.select("#clear-graph-btn");
const pathSourceSelect = d3.select("#path-source");
const pathTargetSelect = d3.select("#path-target");
const findPathBtn = d3.select("#find-path-btn");
const pathResultsDiv = d3.select("#path-results");

// D3 selections for analysis buttons
const centralityBtn = d3.select("#centrality-btn");
const communitiesBtn = d3.select("#communities-btn");
const anomaliesBtn = d3.select("#anomalies-btn");
const predictLinksBtn = d3.select("#predict-links-btn");

// --- Graph Visualization Setup ---
let svg, container, link, node, nodeElements; // Define these globally within the script scope

function setupGraphVisualization() {
    const width = graphContainer.node().getBoundingClientRect().width;
    const height = Math.max(300, graphContainer.node().getBoundingClientRect().height || window.innerHeight * 0.4);

    graphContainer.select("svg").remove();

    svg = graphContainer.append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("max-width", "100%")
        .style("height", "auto");

    container = svg.append("g");

    svg.call(d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 8])
        .on("zoom", ({ transform }) => {
            container.attr("transform", transform);
        }));

    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "-0 -5 10 10")
        // *** Reduce refX slightly: distance FROM the attachment point INTO the marker ***
        .attr("refX", 8) // Point of the arrow will be 8 units along the path from the end point
        .attr("refY", 0)
        .attr("orient", "auto-start-reverse") // Tries to align with the start of the last segment
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("xoverflow", "visible")
        .append("svg:path")
        .attr("d", "M 0,-5 L 10 ,0 L 0,5")
        .attr("fill", "#999")
        .style("stroke", "none");

    // Initialize selections for links (NOW PATHS) and nodes
    link = container.append("g")
        .attr("class", "links")
        .attr("fill", "none") // Paths are filled by default, we want stroke
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 1.5)
        // *** Select PATH instead of line ***
        .selectAll("path");

    node = container.append("g")
        .attr("class", "nodes")
        .selectAll("g");

    simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(100).strength(0.4)) // Slightly reduced strength maybe
        .force("charge", d3.forceManyBody().strength(-180)) // Slightly increased repulsion maybe
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(15)) // Simple collision radius
        // *** Add/Increase Velocity Decay (Friction) ***
        .velocityDecay(0.5) // Default is 0.4, higher means more friction
        .on("tick", ticked);
}

function ticked() {
    // ----- Calculate Path 'd' attribute for curves AND arrow positioning -----
    link.attr("d", d => {
        const source = d.source;
        const target = d.target;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dr = Math.sqrt(dx*dx + dy*dy);

        // If nodes are too close, draw a straight line to avoid weird curves/marker issues
        if (dr < 15) { // Threshold distance
             // Ensure marker doesn't overlap node if line is very short
            if (dr < 1) return `M${source.x},${source.y} L${target.x},${target.y}`; // Avoid division by zero
             const normX = dx / dr;
             const normY = dy / dr;
             // Target point slightly before the actual node center
             const targetX = target.x - normX * 8; // 8 is node radius + buffer
             const targetY = target.y - normY * 8;
             return `M${source.x},${source.y} L${targetX},${targetY}`;
         }


        // Midpoint for control point calculation
        const mx = source.x + dx / 2;
        const my = source.y + dy / 2;

        // Offset perpendicular to the line for the control point
        const curveIntensity = 25; // Adjust for more/less curve
        const offsetX = -(dy / dr) * curveIntensity;
        const offsetY = (dx / dr) * curveIntensity;

        const controlX = mx + offsetX;
        const controlY = my + offsetY;

        // Calculate a point slightly before the target for the marker to attach cleanly
        // We need the tangent at the end of the quadratic curve.
        // Tangent direction vector T = (P1 - C) where P1 is target, C is control point
        const tanX = target.x - controlX;
        const tanY = target.y - controlY;
        const tanLen = Math.sqrt(tanX*tanX + tanY*tanY);

        if (tanLen < 1) { // Avoid division by zero if target equals control point
             return `M${source.x},${source.y} Q${controlX},${controlY} ${target.x},${target.y}`;
        }

        // Normalized tangent vector
        const normTanX = tanX / tanLen;
        const normTanY = tanY / tanLen;

        // How far back from the target to end the visible path line (radius + marker size)
        const endOffset = 8; // Adjust as needed (should be >= node radius)
        const targetXAdjusted = target.x - normTanX * endOffset;
        const targetYAdjusted = target.y - normTanY * endOffset;


        // Quadratic Bezier curve ending slightly before the target center
        return `M${source.x},${source.y} Q${controlX},${controlY} ${targetXAdjusted},${targetYAdjusted}`;
    });

    node.attr("transform", d => `translate(${d.x},${d.y})`);
}


// Drag behavior
function drag(simulation) {
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; // Let the simulation take over again
        d.fy = null;
    }
    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

// --- Update Graph Visualization ---
function updateGraphVisualization() {
    console.log("updateGraphVisualization called. Data:", JSON.stringify(graphData));
    if (!simulation || !svg) return;

    const nodes = graphData.nodes.map(d => ({ ...d }));
    const links = graphData.links.map(d => ({ ...d }));

    // Update simulation nodes & links
    // Important: Need to find node objects for source/target if graphData only has IDs
     const nodeById = new Map(nodes.map(d => [d.id, d]));

    links.forEach(link => {
       link.source = nodeById.get(link.source.id || link.source);
       link.target = nodeById.get(link.target.id || link.target);
     });

     // Filter out links where source or target doesn't exist (can happen during rapid updates)
    const validLinks = links.filter(l => l.source && l.target);

    simulation.nodes(nodes);
    simulation.force("link").links(validLinks);


    // ----- Update Links (Paths) -----
    link = link.data(validLinks, d => `${d.source.id}-${d.target.id}`); // Link identifier
    link.exit().remove();
    link = link.enter().append("path")
        .attr("stroke-width", 1.5)
        // *** Apply marker-end correctly ***
        .attr("marker-end", "url(#arrowhead)")
        .merge(link); // Ensure existing paths also get the marker if it was missing


    // ----- Update Nodes (Groups: Circle + Text) -----
    node = node.data(nodes, d => d.id);
    node.exit().remove();

    const nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .call(drag(simulation))
        .on("click", (event, d) => {
            event.stopPropagation();
            if (d.id !== currentNode) {
                loadPage(d.id);
            } else {
                console.log("Clicked current node:", d.id);
            }
        })
        .on("mouseover", (event, d) => {
            d3.select(event.currentTarget).select('text').style("font-weight", "bold");
             // Optional: highlight neighbors
        })
        .on("mouseout", (event, d) => {
            const textElement = d3.select(event.currentTarget).select('text');
            if (!textElement.classed('current-node-text')) {
                textElement.style("font-weight", "normal");
            }
            // Optional: remove neighbor highlight
        });

    nodeEnter.append("circle")
        .attr("r", 6)
        .attr("fill", "#ccc"); // Default fill

    nodeEnter.append("text")
        .attr("dy", "0.31em")
        .attr("x", 10)
        .style("font-size", "10px")
        .style("fill", "#333")
        .classed("noselect", true)
        .text(d => d.label.length > 20 ? d.label.substring(0, 18) + "..." : d.label);

    node = nodeEnter.merge(node);
    nodeElements = container.selectAll(".node"); // Update selection

    // --- Highlight current node ---
    node.select("circle")
        .attr("r", d => d.id === currentNode ? 8 : 6)
        .attr("fill", d => d.id === currentNode ? "#0d6efd" : "#ccc");

    node.select("text")
        .classed("current-node-text", d => d.id === currentNode)
        .style("font-weight", d => d.id === currentNode ? "bold" : "normal");


    // Restart simulation: Use lower alphaTarget after updates to reduce initial chaos
    // *** Adjust alphaTarget for stabilization ***
    simulation.alphaTarget(0.05).restart(); // Lower target alpha (e.g., 0.05 or 0.1)
    // You might even skip setting alphaTarget and just use restart() if velocityDecay is high enough
    // simulation.restart();
}


// --- API Interaction ---

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Fetch graph data from backend
function fetchGraphData(callback) {
    graphLoadingSpinner.style("display", "block");
    fetch('/api/graph')
        .then(response => response.json())
        .then(data => {
            graphData = data;
            if (callback) callback();
            updateGraphVisualization();
            updatePathSelectors(); // Update dropdowns after graph data is fetched
        })
        .catch(error => console.error('Error fetching graph data:', error))
        .finally(() => graphLoadingSpinner.style("display", "none"));
}

// Load Wikipedia page
function loadPage(title) {
    if (!title) return;
    title = title.trim();
    if (!title) return;

    console.log(`Loading page: ${title}, Mode: ${currentMode}`);
    contentTitle.text(`Loading ${title}...`);
    contentBody.html('<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
    analysisResultsContainer.html("").style("display", "none"); // Clear analysis results

    // Fetch page data from backend API
    fetch(`/api/page/${encodeURIComponent(title)}?mode=${currentMode}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    // Handle disambiguation specifically
                    if (response.status === 400 && err.is_disambiguation) {
                        throw { is_disambiguation: true, message: err.error, options: err.options };
                    }
                     throw new Error(err.error || `HTTP error ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            currentNode = data.title; // Update the current node
            contentTitle.text(data.title);
            // Load actual Wikipedia content into an iframe for safe rendering
            contentBody.html(`<iframe id="wiki-iframe" src="/wikipedia_html/${encodeURIComponent(data.title)}"></iframe>`);
            // Fetch updated graph data AFTER the page has been processed by the backend
            fetchGraphData();
            // Add the loaded title to the search input for clarity
            searchInput.property("value", data.title);
            suggestionsList.html(''); // Clear suggestions
        })
        .catch(error => {
            console.error('Error loading page:', error);
             contentTitle.text("Error");
             if (error.is_disambiguation) {
                  let optionsHtml = error.options.slice(0, 10).map(opt =>
                     `<li><a href="#" class="disambiguation-link">${opt}</a></li>`
                 ).join('');
                  contentBody.html(`<div class="alert alert-warning"><strong>Disambiguation Error:</strong> "${searchInput.property("value")}" may refer to:<ul>${optionsHtml}</ul></div>`);
                 // Add listeners to the new links
                 contentBody.selectAll(".disambiguation-link").on("click", function(event) {
                     event.preventDefault();
                     loadPage(d3.select(this).text());
                 });
             } else {
                 contentBody.html(`<div class="alert alert-danger">${error.message || 'Failed to load page.'}</div>`);
             }
            // Fetch graph data even on error to show current state
            fetchGraphData();
        });
}

// Search Wikipedia for suggestions
const searchWiki = debounce(() => {
    const query = searchInput.property("value").trim();
    if (query.length < 2) {
        suggestionsList.html('');
        return;
    }

    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            let suggestionsHtml = '';
            if (data.results && data.results.length > 0) {
                suggestionsHtml = data.results.map(title =>
                    `<button type="button" class="list-group-item list-group-item-action suggestion-item">${title}</button>`
                ).join('');
            } else {
                suggestionsHtml = '<span class="list-group-item">No suggestions found</span>';
            }
            suggestionsList.html(suggestionsHtml);

            // Add click listener to suggestions
            d3.selectAll(".suggestion-item").on("click", function() {
                const selectedTitle = d3.select(this).text();
                searchInput.property("value", selectedTitle); // Update input field
                suggestionsList.html(''); // Clear suggestions
                loadPage(selectedTitle); // Load the selected page
            });
        })
        .catch(error => {
            console.error("Error searching Wikipedia:", error);
            suggestionsList.html('<span class="list-group-item text-danger">Search error</span>');
        });
}, 300); // Debounce time in ms

// Load Random Article
function loadRandomArticle() {
     contentTitle.text("Loading Random Article...");
     contentBody.html('<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
     analysisResultsContainer.html("").style("display", "none"); // Clear analysis results

     fetch("https://en.wikipedia.org/api/rest_v1/page/random/title")
         .then(response => {
              if (!response.ok) throw new Error(`HTTP error ${response.status}`);
              return response.json();
          })
         .then(data => {
              if (data.items && data.items.length > 0) {
                  const randomTitle = data.items[0].title.replace(/_/g, ' '); // Ensure spaces
                  loadPage(randomTitle);
              } else {
                  throw new Error("No random article title received.");
              }
          })
         .catch(error => {
              console.error("Error fetching random article:", error);
              contentTitle.text("Error");
              contentBody.html('<div class="alert alert-warning">Could not fetch random article. Please try searching.</div>');
              // Fallback: load a default page?
              // loadPage("Social network analysis");
          });
 }


// Clear Graph
function clearGraph() {
    if (autoRefreshIntervalId) {
        clearInterval(autoRefreshIntervalId);
        autoRefreshIntervalId = null;
        console.log("clearGraph: Stopped auto-refresh interval.");
    }

    if (confirm("Are you sure you want to clear the entire graph? This cannot be undone.")) {
        fetch('/api/clear_graph', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                graphData = { nodes: [], links: [] }; // Clear local data
                currentNode = null;
                updateGraphVisualization();
                updatePathSelectors(); // Clear dropdowns
                contentTitle.text("Graph Cleared");
                contentBody.html("<p>Graph has been cleared. Search for a page to start again.</p>");
                analysisResultsContainer.html("").style("display", "none");
                searchInput.property("value", ""); // Clear search input
            })
            .catch(error => {
                console.error('Error clearing graph:', error);
                alert("Failed to clear graph. See console for details.");
            });
    }
}

// Update Path Selectors
function updatePathSelectors() {
    const nodes = graphData.nodes.map(d => d.id).sort();

    // Preserve selected values if they still exist
    const currentSource = pathSourceSelect.property("value");
    const currentTarget = pathTargetSelect.property("value");

    pathSourceSelect.html('<option selected disabled value="">Select source node</option>'); // Add placeholder
    pathTargetSelect.html('<option selected disabled value="">Select target node</option>'); // Add placeholder

    nodes.forEach(nodeId => {
        pathSourceSelect.append("option").attr("value", nodeId).text(nodeId);
        pathTargetSelect.append("option").attr("value", nodeId).text(nodeId);
    });

     // Restore selection if possible
    if (nodes.includes(currentSource)) {
         pathSourceSelect.property("value", currentSource);
     }
    if (nodes.includes(currentTarget)) {
         pathTargetSelect.property("value", currentTarget);
     }

    // Optional: Set source to current node if available
    if (currentNode && nodes.includes(currentNode) && !pathSourceSelect.property("value")) {
        pathSourceSelect.property("value", currentNode);
    }
}

// Find Shortest Path
function findShortestPath() {
    const source = pathSourceSelect.property("value");
    const target = pathTargetSelect.property("value");

     pathResultsDiv.html(""); // Clear previous results
     // Clear highlights
     link?.classed("path-highlight", false);
     nodeElements?.classed("node-highlight", false);
     nodeElements?.select("text").classed("node-highlight", false); // Ensure text highlight is removed too


    if (!source || !target || source === target) {
        pathResultsDiv.html('<div class="alert alert-warning p-2">Please select distinct source and target nodes.</div>');
        return;
    }

    pathResultsDiv.html('<div class="spinner-border spinner-border-sm" role="status"><span class="visually-hidden">Loading...</span></div> Finding path...');

    fetch(`/api/shortest_path?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`)
        .then(response => {
            if (!response.ok) {
                 return response.json().then(err => { throw new Error(err.error || `HTTP error ${response.status}`) });
            }
            return response.json();
        })
        .then(data => {
            let pathHtml = `<p class="mb-1">Path (${data.length} steps):</p><ol class="list-group list-group-numbered list-group-flush">`;
            data.path.forEach(nodeName => {
                 // Make list items clickable to load the page
                 pathHtml += `<li class="list-group-item p-1 border-0"><a href="#" class="path-node-link" data-node="${nodeName}">${nodeName}</a></li>`;
             });
            pathHtml += "</ol>";
            pathResultsDiv.html(pathHtml);

            // Add click handlers to path nodes
            pathResultsDiv.selectAll(".path-node-link").on("click", function(event) {
                 event.preventDefault();
                 const nodeName = d3.select(this).attr("data-node");
                 loadPage(nodeName);
            });

            // Highlight path in the graph
            link?.classed("path-highlight", false); // Reset previous highlights
            nodeElements?.classed("node-highlight", false);
            nodeElements?.select("text").classed("node-highlight", false);


            data.path_data.forEach(edge => {
                 link?.filter(d => (d.source.id === edge.source || d.source === edge.source) && (d.target.id === edge.target || d.target === edge.target))
                     .classed("path-highlight", true)
                     .raise(); // Bring highlighted links to front
             });

            data.path.forEach(nodeName => {
                 nodeElements?.filter(d => d.id === nodeName)
                     .classed("node-highlight", true)
                     .select("text").classed("node-highlight", true); // Style text too
                  nodeElements?.filter(d => d.id === nodeName).raise(); // Bring nodes to front
             });
        })
        .catch(error => {
            console.error("Error finding path:", error);
            pathResultsDiv.html(`<div class="alert alert-danger p-2">${error.message || 'Failed to find path.'}</div>`);
        });
}

// --- Analysis Functions ---

function displayAnalysisResults(title, htmlContent) {
     contentTitle.text(title); // Update main title to reflect analysis
     contentBody.html(""); // Clear the iframe/welcome message
     analysisResultsContainer.html(htmlContent).style("display", "block");

     // Make table links clickable
     analysisResultsContainer.selectAll(".analysis-page-link").on("click", function(event) {
         event.preventDefault();
         const nodeName = d3.select(this).attr("data-node");
         loadPage(nodeName);
     });

     analysisResultsContainer.selectAll(".community-toggle").on("click", function() {
        const targetId = d3.select(this).attr("data-bs-target");
        const isExpanded = d3.select(this).attr("aria-expanded") === "true";
        // Optional: Change icon based on state
        const icon = d3.select(this).select("i.toggle-icon");
        if (icon) {
            icon.classed("fa-chevron-down", isExpanded).classed("fa-chevron-right", !isExpanded);
        }
    });
     // --- Add event listener for member links inside communities ---
      analysisResultsContainer.selectAll(".community-member-link").on("click", function(event) {
         event.preventDefault();
         const nodeName = d3.select(this).attr("data-node");
         loadPage(nodeName);
     });

      // Add handlers for prediction verification if needed
     analysisResultsContainer.selectAll(".verify-link-btn").on("click", function(event) {
         event.preventDefault();
          const source = d3.select(this).attr("data-source");
         const target = d3.select(this).attr("data-target");
         // Action: Load source page, maybe highlight target links?
          loadPage(source);
          // Could post message to iframe to find/highlight links to target after load
          setTimeout(() => {
                const iframe = d3.select("#wiki-iframe").node();
                if (iframe && iframe.contentWindow) {
                    console.log(`Attempting to highlight links to ${target} in iframe`);
                     iframe.contentWindow.postMessage({ type: 'highlightLink', target: target }, '*');
                     // The iframe needs a listener for this message
                 }
           }, 2500); // Delay to allow iframe loading
      });
 }


function runCentralityAnalysis() {
    displayAnalysisResults("Centrality Analysis", '<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div> Calculating...</div>');
     nodeElements?.classed("highlight-yellow", false); // Clear other highlights
     link?.classed("path-highlight", false);
     nodeElements?.classed("node-highlight", false);
     nodeElements?.select("text").classed("node-highlight", false);

    fetch("/api/centrality")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                displayAnalysisResults("Centrality Analysis", '<div class="alert alert-info">Not enough data in the graph for centrality analysis.</div>');
                return;
            }
            let tableHtml = `
                <p>Top nodes ranked by PageRank:</p>
                <div class="table-responsive">
                <table class="table table-sm table-striped table-hover">
                    <thead>
                        <tr><th>Rank</th><th>Page</th><th>PageRank</th><th>In-Degree</th><th>Out-Degree</th><th>Betweenness</th></tr>
                    </thead>
                    <tbody>`;
            data.slice(0, 25).forEach((item, index) => { // Show top 25
                tableHtml += `
                    <tr>
                        <td>${index + 1}</td>
                        <td><a href="#" class="analysis-page-link" data-node="${item.node}">${item.node}</a></td>
                        <td>${item.pagerank.toFixed(5)}</td>
                        <td>${item.in_degree}</td>
                        <td>${item.out_degree}</td>
                        <td>${item.betweenness.toFixed(5)}</td>
                    </tr>`;
            });
            tableHtml += `</tbody></table></div>`;
            displayAnalysisResults("Centrality Analysis", tableHtml);
        })
        .catch(error => {
             console.error("Centrality Analysis Error:", error);
             displayAnalysisResults("Centrality Analysis", `<div class="alert alert-danger">Error calculating centrality: ${error.message}</div>`);
         });
}

function runCommunityDetection() {
    displayAnalysisResults("Community Detection", '<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div> Detecting communities...</div>');
    // ... (clear highlights) ...

    fetch("/api/communities")
        .then(response => response.json())
        .then(data => {
            if (!data || !data.communities || data.communities.length === 0) {
                // ... (handle no communities found) ...
                return;
            }

            // --- Generate Community Names (Heuristic) ---
            // Simple heuristic: Use the top 1-3 most central nodes (e.g., by degree or PageRank within the community)
            // This requires centrality data, which we don't have per-community here easily.
            // Alternative: Just use "Community X" or the first few member names.
            // Let's use the first 1-2 member names for simplicity.
            function getCommunityName(nodes) {
                if (!nodes || nodes.length === 0) return "Unnamed Community";
                if (nodes.length === 1) return nodes[0];
                // Simple name based on first two nodes
                 return `${nodes[0].substring(0, 15)} / ${nodes[1].substring(0, 15)} Cluster`; // Truncate long names
                 // More advanced: Fetch degrees/pagerank for nodes *within the current graph* and pick top ones
            }

            // --- Build Collapsible Accordion HTML ---
            let communitiesHtml = `<p>Detected ${data.communities.length} communities:</p>
                 <div class="accordion" id="communitiesAccordion">`;

            data.communities.forEach((comm, index) => {
                const communityId = comm.id;
                const communityNodes = comm.nodes;
                const collapseId = `community-collapse-${communityId}`;
                const headerId = `community-header-${communityId}`;
                const communityName = getCommunityName(communityNodes); // Generate a name
                const colorClass = `community-${communityId % 10}`; // Cycle through 10 colors

                communitiesHtml += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="${headerId}">
                        <button class="accordion-button collapsed community-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
                            <span class="badge rounded-pill me-2" style="background-color: var(--bs-${colorClass}-color, #6c757d); color: white;">${communityId}</span>
                             ${communityName}
                            <span class="ms-auto me-2 text-muted">(${communityNodes.length} members)</span>
                            <i class="fas fa-chevron-right toggle-icon ms-1"></i>  <!-- Icon for visual cue -->
                        </button>
                    </h2>
                    <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headerId}" data-bs-parent="#communitiesAccordion">
                        <div class="accordion-body">
                            <ul class="list-group list-group-flush">`;

                // Add members as clickable links inside the collapsible body
                communityNodes.forEach(nodeName => {
                    communitiesHtml += `<li class="list-group-item py-1"><a href="#" class="community-member-link" data-node="${nodeName}">${nodeName}</a></li>`;
                });

                communitiesHtml += `
                            </ul>
                        </div>
                    </div>
                </div>`;
            });

            communitiesHtml += `</div>`; // End accordion

            displayAnalysisResults("Community Detection", communitiesHtml);

            // Color nodes in the graph (same logic as before)
            const nodeMap = new Map(data.node_communities.map(item => [item.node, item.community]));
            nodeElements?.select("circle")
                .attr("class", d => `node-circle community-${(nodeMap.get(d.id) ?? -1) % 10}-fill`)
                .style("fill", d => {
                    const commId = nodeMap.get(d.id);
                    if (commId !== undefined && commId !== -1) {
                        const colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395"];
                        return colors[commId % colors.length];
                    }
                    return "#ccc";
                });

        })
        .catch(error => {
            console.error("Community Detection Error:", error);
            displayAnalysisResults("Community Detection", `<div class="alert alert-danger">Error detecting communities: ${error.message}</div>`);
        });
}


function runAnomalyDetection() {
     displayAnalysisResults("Anomaly Detection", '<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div> Finding anomalies...</div>');
     nodeElements?.select("circle").style("fill", null).attr("class", "node-circle"); // Reset community colors
     link?.classed("path-highlight", false);
     nodeElements?.classed("node-highlight", false);
     nodeElements?.select("text").classed("node-highlight", false);

    fetch("/api/anomalies")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                displayAnalysisResults("Anomaly Detection", '<div class="alert alert-info">No anomalies detected based on current heuristics.</div>');
                nodeElements?.classed("highlight-yellow", false); // Ensure highlight is off
                return;
            }

            let anomaliesHtml = `<p>Detected ${data.length} potentially anomalous nodes:</p>
                 <div class="table-responsive">
                 <table class="table table-sm table-striped table-hover">
                     <thead><tr><th>Page</th><th>Type(s)</th><th>In</th><th>Out</th><th>PageRank</th></tr></thead>
                     <tbody>`;
             const anomalousNodes = [];
             data.forEach(item => {
                 anomalousNodes.push(item.node);
                 const types = item.type.join(', ').replace(/_/g, ' '); // Make types more readable
                 anomaliesHtml += `<tr>
                                     <td><a href="#" class="analysis-page-link" data-node="${item.node}">${item.node}</a></td>
                                     <td><span class="badge bg-warning text-dark">${types}</span></td>
                                     <td>${item.in_degree}</td>
                                     <td>${item.out_degree}</td>
                                     <td>${item.pagerank.toFixed(5)}</td>
                                  </tr>`;
             });
             anomaliesHtml += `</tbody></table></div>`;
             displayAnalysisResults("Anomaly Detection", anomaliesHtml);

             // Highlight anomalous nodes in yellow
             nodeElements?.classed("highlight-yellow", d => anomalousNodes.includes(d.id));

         })
         .catch(error => {
              console.error("Anomaly Detection Error:", error);
              displayAnalysisResults("Anomaly Detection", `<div class="alert alert-danger">Error detecting anomalies: ${error.message}</div>`);
         });
}

function runLinkPrediction() {
     displayAnalysisResults("Link Prediction", '<div class="loading"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div> Predicting links...</div>');
     nodeElements?.classed("highlight-yellow", false); // Clear other highlights
     nodeElements?.select("circle").style("fill", null).attr("class", "node-circle");
     link?.classed("path-highlight", false);
     nodeElements?.classed("node-highlight", false);
     nodeElements?.select("text").classed("node-highlight", false);

    fetch("/api/predict_links")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                displayAnalysisResults("Link Prediction", '<div class="alert alert-info">Not enough data or no potential links found with current heuristic.</div>');
                return;
            }

             let predictionsHtml = `<p>Top potential links based on common neighbors:</p>
                 <div class="table-responsive">
                 <table class="table table-sm table-striped table-hover">
                      <thead><tr><th>Node 1</th><th>Node 2</th><th>Score (Common Neighbors)</th><th>Action</th></tr></thead>
                     <tbody>`;
             data.forEach(item => {
                 predictionsHtml += `<tr>
                                      <td><a href="#" class="analysis-page-link" data-node="${item.u}">${item.u}</a></td>
                                      <td><a href="#" class="analysis-page-link" data-node="${item.v}">${item.v}</a></td>
                                      <td>${item.score}</td>
                                      <td><button class="btn btn-sm btn-outline-primary verify-link-btn" data-source="${item.u}" data-target="${item.v}">Verify ${item.u} → ${item.v}</button></td>
                                   </tr>`;
                                   // Add button for other direction?
                                   // <td><button class="btn btn-sm btn-outline-primary verify-link-btn" data-source="${item.v}" data-target="${item.u}">Verify ${item.v} → ${item.u}</button></td>
             });
             predictionsHtml += `</tbody></table></div>`;
             displayAnalysisResults("Link Prediction", predictionsHtml);
         })
         .catch(error => {
              console.error("Link Prediction Error:", error);
              displayAnalysisResults("Link Prediction", `<div class="alert alert-danger">Error predicting links: ${error.message}</div>`);
         });
 }


// --- Event Listeners ---
function setupEventListeners() {
    // Search
    searchBtn.on("click", () => loadPage(searchInput.property("value")));
    searchInput.on("keypress", (event) => {
        if (event.key === "Enter") {
            suggestionsList.html(''); // Clear suggestions immediately on enter
            loadPage(searchInput.property("value"));
        }
    });
    searchInput.on("input", searchWiki); // Use debounced search for suggestions

    // Random Article
    randomArticleBtn.on("click", loadRandomArticle);

    // Mode Toggle
    // --- Modify Mode Toggle Listener ---
    modeToggle.on("change", function() {
        const isChecked = d3.select(this).property("checked");
        const newMode = isChecked ? 'extract_links' : 'add_node';

        console.log(`Mode changed to: ${newMode}`);

        // *** Clear existing auto-refresh interval ***
        if (autoRefreshIntervalId) {
            clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = null;
            console.log("Stopped auto-refresh interval due to mode change.");
        }

        // Update global mode variable and UI text
        currentMode = newMode;
        modeLabel.text(currentMode === 'extract_links' ? "Extract Links Mode" : "Add Node Only Mode");
        const descText = currentMode === 'extract_links'
            ? "<strong>Current: Extract Links Mode</strong>..."
            : "<strong>Current: Add Node Only Mode</strong>...";
        modeDescription.html(descText);


        // *** Start interval ONLY if the new mode is 'add_node' ***
        if (currentMode === 'add_node') {
            console.log(`Starting auto-refresh interval (${AUTO_REFRESH_INTERVAL_MS / 1000} seconds) for Add Node Only mode.`);
            autoRefreshIntervalId = setInterval(() => {
                // Check the flag before fetching inside the interval
                fetchGraphData();
            }, AUTO_REFRESH_INTERVAL_MS); // Use the constant for the interval
        }
    });
    // --- End Mode Toggle Listener Modification ---

    // Clear Graph
    clearGraphBtn.on("click", clearGraph);

    // Path Finding
    findPathBtn.on("click", findShortestPath);

    // Analysis Buttons
    centralityBtn.on("click", runCentralityAnalysis);
    communitiesBtn.on("click", runCommunityDetection);
    anomaliesBtn.on("click", runAnomalyDetection);
    predictLinksBtn.on("click", runLinkPrediction);

    // Window resize handler
    window.addEventListener('resize', debounce(() => {
        console.log("Resizing graph...");
        setupGraphVisualization(); // Reinitialize SVG size and simulation center
        fetchGraphData(); // Refetch and redraw with current data
    }, 500)); // Debounce resize events

     // Listener for messages from iframe (e.g., for link clicking, highlighting)
     window.addEventListener('message', (event) => {
        // Basic security check (optional but good practice)
        // if (event.origin !== window.location.origin) {
        //     console.warn("Ignoring message from potentially untrusted origin:", event.origin);
        //     return;
        // }

        // Check if the data is a string and starts with our prefix
        if (typeof event.data === 'string' && event.data.startsWith('loadPage:')) {
            // Extract the page title after the prefix
            const pageToLoad = event.data.substring('loadPage:'.length);
            console.log("Received 'loadPage' message from iframe for:", pageToLoad);
            if (pageToLoad) {
                loadPage(pageToLoad); // Call your existing function
            }
        }
        // You could add handlers for other message types here if needed
        // else if (event.data.type === 'otherAction') { ... }
    });
}

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    setupGraphVisualization();
    setupEventListeners();

    // Initial check: If starting in add_node mode (e.g., checkbox is unchecked by default), start the interval
    currentMode = modeToggle.property("checked") ? 'extract_links' : 'add_node';
    if (currentMode === 'add_node') {
         console.log(`Initial state: Starting auto-refresh interval (${AUTO_REFRESH_INTERVAL_MS / 1000} seconds).`);
         autoRefreshIntervalId = setInterval(() => {
             if (!isGraphOperationInProgress) {
                 console.log(`Auto-refresh (${new Date().toLocaleTimeString()}): Fetching graph data...`);
                 fetchGraphData();
             } else {
                 console.log(`Auto-refresh (${new Date().toLocaleTimeString()}): Skipped fetch, graph operation in progress.`);
             }
         }, AUTO_REFRESH_INTERVAL_MS);
     }

    // Update initial mode description
     const initialDescText = currentMode === 'extract_links' ? "<strong>Current: Extract Links Mode</strong>..." : "<strong>Current: Add Node Only Mode</strong>...";
     modeDescription.html(initialDescText);


    fetchGraphData(); // Load initial graph state
});