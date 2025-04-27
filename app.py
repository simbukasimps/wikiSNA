from flask import Flask, request, jsonify, render_template, Response
import networkx as nx
import wikipedia
import json
import re
import community as community_louvain
# from sklearn.metrics import silhouette_score # Not used in current endpoints
from collections import defaultdict
import numpy as np
# from datetime import datetime # Not used
import requests
# from bs4 import BeautifulSoup # Not used directly here anymore
# import os # Not used for graph persistence anymore
# import pickle # Not used for graph persistence anymore
import logging
import threading # Import threading
import time # For potential delays or logging

app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.INFO)

# Initialize the graph in memory
G = nx.DiGraph()
graph_lock = threading.Lock()

# Cache for Wikipedia pages (simple in-memory)
page_cache = {}
# Cache for Wikipedia HTML content
html_cache = {}
# *** NEW: Cache for link sets {title: set(links)} ***
link_cache = {}

# --- Graph Construction Modes ---
MODE_EXTRACT_LINKS = "extract_links"
MODE_ADD_NODE_ONLY = "add_node"
# --- ---

# === Helper Function for Cached Link Fetching ===
def get_links_with_cache(page_title):
    """
    Fetches the set of links for a Wikipedia page, using a cache.
    Returns the set of links, or None if the page cannot be fetched or has no links.
    Caches the result (including errors as None or empty set).
    """
    if page_title in link_cache:
        # app.logger.debug(f"Link cache HIT for: {page_title}")
        return link_cache[page_title] # Return cached set (could be None or empty)

    app.logger.info(f"Link cache MISS for: {page_title}. Fetching from Wikipedia...")
    try:
        page = wikipedia.page(page_title, auto_suggest=False, preload=False) # Avoid preloading content if only links are needed
        links = set(page.links)
        link_cache[page_title] = links # Cache the successful result
        return links
    except wikipedia.exceptions.PageError:
        app.logger.warning(f"PageNotFound when fetching links for cache: {page_title}")
        link_cache[page_title] = None # Cache the error state (page doesn't exist)
        return None
    except wikipedia.exceptions.DisambiguationError:
        app.logger.warning(f"DisambiguationError when fetching links for cache: {page_title}")
        link_cache[page_title] = None # Cache the error state
        return None
    except Exception as e:
        # Catch other potential errors (network issues, etc.)
        app.logger.error(f"Error fetching links for '{page_title}': {e}")
        link_cache[page_title] = None # Cache general error state
        return None
# === End Helper Function ===

def check_backlinks_background(target_title, nodes_to_check):
    """
    Checks for links from 'nodes_to_check' pointing to 'target_title'.
    Runs in a background thread and updates the global graph G safely.
    """
    app.logger.info(f"[BG Task {target_title}] Starting backlink check from {len(nodes_to_check)} nodes.")
    start_time = time.time()
    added_count = 0

    for existing_node in nodes_to_check:
        # No need to check self -> self
        if existing_node == target_title:
            continue

        # Get links (using cache) - this might block if uncached
        existing_node_links = get_links_with_cache(existing_node)

        if existing_node_links is not None and target_title in existing_node_links:
            # Acquire lock before modifying the shared graph
            with graph_lock:
                if not G.has_edge(existing_node, target_title):
                    app.logger.debug(f"[BG Task {target_title}] Adding backlink: {existing_node} -> {target_title}")
                    G.add_edge(existing_node, target_title)
                    added_count += 1
        # Optional: Small sleep to yield CPU if task is long
        # time.sleep(0.01)

    end_time = time.time()
    duration = end_time - start_time
    app.logger.info(f"[BG Task {target_title}] Finished backlink check in {duration:.2f}s. Added {added_count} edges.")
    # Here you *could* potentially signal the frontend via WebSockets/SSE if implemented.

# === End Background Task ===


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

# --- Combined Page Loading Endpoint ---
@app.route('/api/page/<title>')
def get_page(title):
    """
    Fetches a Wikipedia page and updates the graph based on the requested mode.
    Uses caches for HTML and link sets.
    """
    title = title.replace('_', ' ')
    mode = request.args.get('mode', MODE_EXTRACT_LINKS)

    # --- Fetch HTML (use cache first) ---
    if title in html_cache: # ...
         html_content = html_cache[title] # ...
         try: # ...
             page = wikipedia.page(title, auto_suggest=False, preload=False) # ...
             pageid = page.pageid # ...
         except Exception: pageid = None # ...
    else: # ...
         try: # ...
             page = wikipedia.page(title, auto_suggest=False) # ...
             html_content = page.html() # ...
             pageid = page.pageid # ...
             html_cache[title] = html_content # ...
         except wikipedia.exceptions.PageError: return jsonify({'error': ...}), 404 # ...
         except wikipedia.exceptions.DisambiguationError as e: return jsonify({'error': ..., 'options': e.options}), 400 # ...
         except Exception as e: return jsonify({'error': ...}), 500 # ...

    # --- Update Graph based on Mode ---
    try:
        if mode == MODE_EXTRACT_LINKS:
             # Use lock for graph modifications
             with graph_lock:
                if title not in G:
                    G.add_node(title, id=title, pageid=pageid)
                # Fetch links for the title page itself
                title_links = get_links_with_cache(title)
                if title_links is None: title_links = set()

                for link in title_links:
                    if link and not re.match(r'(File:|Image:|Category:|Help:|Wikipedia:|Portal:|Talk:|Template:|Special:)', link, re.IGNORECASE):
                        if link not in G:
                            G.add_node(link, id=link, pageid=None)
                        if not G.has_edge(title, link):
                            G.add_edge(title, link)

        elif mode == MODE_ADD_NODE_ONLY:
            app.logger.info(f"Processing '{title}' in MODE_ADD_NODE_ONLY (Phase 1)...")
            nodes_before_add = list(G.nodes()) # Get existing nodes BEFORE adding the new one

            # --- Phase 1: Quick Update ---
            # Use lock for graph modifications
            with graph_lock:
                # 1. Add/Update the target node
                if title not in G:
                    app.logger.debug(f"Adding node: {title}")
                    G.add_node(title, id=title, pageid=pageid)
                else:
                    if G.nodes[title].get('pageid') is None and pageid is not None:
                        G.nodes[title]['pageid'] = pageid

                # 2. Get links FROM the 'title' page
                title_links = get_links_with_cache(title)
                if title_links is None: title_links = set()

                # 3. Add forward edges FROM 'title' TO existing nodes
                added_forward_links = 0
                for target_node in title_links:
                    if target_node in nodes_before_add and target_node != title: # Check against nodes that existed *before* this call
                        if not G.has_edge(title, target_node):
                            app.logger.debug(f"Adding forward link: {title} -> {target_node}")
                            G.add_edge(title, target_node)
                            added_forward_links += 1
                app.logger.info(f"Added {added_forward_links} immediate forward links from '{title}'.")
            # --- End Phase 1 ---

            # --- Phase 2: Start Background Task for Backlinks ---
            # Create a copy of the node list to check against
            # Include the newly added 'title' node itself in case others link to it
            nodes_to_check_in_background = list(G.nodes()) # Use current graph state

            if nodes_to_check_in_background: # Only start if there are nodes to check
                app.logger.info(f"Starting background task to check backlinks to '{title}'...")
                # Important: Pass copies or immutable data to the thread if necessary
                # Here, title (string) and the list of nodes are okay.
                thread = threading.Thread(
                    target=check_backlinks_background,
                    args=(title, nodes_to_check_in_background),
                    daemon=True # Allows app to exit even if threads are running
                )
                thread.start()
            else:
                 app.logger.info("No existing nodes, skipping background backlink check.")
            # --- End Phase 2 ---

        else: # Invalid mode
             return jsonify({'error': f'Invalid mode: {mode}'}), 400

        # --- Send Response Immediately After Phase 1 ---
        result = {
            'title': title,
            'pageid': pageid,
            'content': html_content
        }
        app.logger.info(f"Sending initial response for '{title}' while background checks may run.")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error during page processing for '{title}': {e}", exc_info=True)
        return jsonify({'error': f'Error processing page request: {str(e)}'}), 500




@app.route('/api/graph')
def get_graph():
    """Return the current graph structure (thread-safe read)."""
    # Acquire lock only if reading complex properties, simple iteration is often okay
    # For safety, especially if nodes/edges might be added/removed during read:
    with graph_lock:
        nodes = [{'id': node, 'label': node, 'pageid': G.nodes[node].get('pageid')}
                 for node in G.nodes()]
        links = [{'source': s, 'target': t} for s, t in G.edges()]

    return jsonify({
        'nodes': nodes,
        'links': links
    })

@app.route('/api/clear_graph', methods=['POST'])
def clear_graph():
    """Clears the graph and all caches (thread-safe)."""
    global G, page_cache, html_cache, link_cache
    # Acquire lock before clearing shared resources
    with graph_lock:
        G = nx.DiGraph()
        page_cache = {}
        html_cache = {}
        link_cache = {}
        app.logger.info("Graph and all caches cleared.")
    return jsonify({'status': 'success', 'message': 'Graph and caches cleared successfully'})

# --- Add wikipedia_html endpoint back ---
@app.route('/wikipedia_html/<path:title>')
def wikipedia_html(title):
    """Fetch Wikipedia HTML and inject a script to handle internal link clicks."""
    title = title.replace('_', ' ')

    # Use html_cache if available
    if title in html_cache:
        html_content = html_cache[title]
    else:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            html_content = page.html()
            html_cache[title] = html_content # Cache it
        except wikipedia.exceptions.PageError:
            return f"<div class='alert alert-danger'>Error: Page '{title}' not found on Wikipedia.</div>", 404
        except wikipedia.exceptions.DisambiguationError as e:
            options_html = "".join([f"<li><a href='#' onclick='window.parent.postMessage(\"loadPage:{opt.replace('_', ' ')}\", \"*\"); return false;'>{opt}</a></li>" for opt in e.options[:10]])
            return f"<div class='alert alert-warning'>Disambiguation Error: '{title}' may refer to:<ul class='list-unstyled'>{options_html}</ul></div>", 400
        except Exception as e:
            app.logger.error(f"Error fetching HTML for '{title}': {e}")
            return f"<div class='alert alert-danger'>Error loading page content: {str(e)}</div>", 500

    # --- Inject JavaScript for Link Interception ---
    # This script runs inside the iframe
    click_interceptor_js = """
    <script>
      document.addEventListener('DOMContentLoaded', function() {
        document.body.addEventListener('click', function(event) {
          // Find the anchor tag the user clicked on, even if they clicked an element inside it
          const link = event.target.closest('a');

          if (link && link.hasAttribute('href')) {
            const href = link.getAttribute('href');
            const title = link.getAttribute('title'); // Sometimes the title attribute is useful

            // Check if it's a standard Wikipedia article link (relative)
            // Regex: Starts with /wiki/, doesn't contain ':', and isn't the Main_Page
            const wikiLinkRegex = /^\/wiki\/([^:]+)$/;
            const match = href.match(wikiLinkRegex);

            // Avoid special pages (like File:, Category:, Special:) or non-article links
            const isSpecialPage = href.includes(':') || href.startsWith('#') || link.classList.contains('external');
            // Check if the title attribute indicates a non-article page (less reliable)
            const isNonArticleTitle = title && /^(Category|File|Portal|Help|Wikipedia|Template|Special|Talk):/.test(title);

            if (match && match[1] && match[1].toLowerCase() !== 'main_page' && !isSpecialPage && !isNonArticleTitle) {
              event.preventDefault(); // Stop the browser from navigating
              event.stopPropagation(); // Stop the event bubbling up further

              // Extract the article title (decode URI components and replace underscores)
              let targetTitle = decodeURIComponent(match[1]).replace(/_/g, ' ');

              console.log('Intercepted Wiki Link Click:', targetTitle);

              // Send message to parent window to load the page
              // Format: "loadPage:<Actual Page Title>"
              window.parent.postMessage('loadPage:' + targetTitle, '*'); // '*' allows any parent origin, okay for local dev
            } else {
              // Optional: Handle external links or special links differently if needed
              // For external links, make sure they open in a new tab
              if (link.classList.contains('external') || href.startsWith('http://') || href.startsWith('https://')) {
                  link.target = '_blank'; // Force external links or non-intercepted links to open new tab
              }
               // console.log('Ignoring non-article link:', href);
            }
          }
        });
      });
    </script>
    """
    # Append the script to the body (or just before </body> if possible)
    # A simple append might be sufficient for most cases
    processed_html = html_content + click_interceptor_js

    return Response(processed_html, mimetype='text/html')

# --- Search Endpoint (Keep as is) ---
@app.route('/api/search')
def search_wikipedia():
    """Search Wikipedia for pages matching a query."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': []})

    try:
        # Search Wikipedia
        search_results = wikipedia.search(query, results=10)
        return jsonify({'results': search_results})
    except Exception as e:
        app.logger.error(f"Error during Wikipedia search for '{query}': {e}")
        return jsonify({'error': f'Error during search: {str(e)}'}), 500

# --- Analysis Endpoints (Keep as they are, ensure they handle empty graphs) ---

@app.route('/api/centrality')
def get_centrality():
    """Compute and return centrality metrics for nodes in the graph."""
    if len(G) == 0:
        # Return empty list instead of error for better frontend handling
        return jsonify([])

    try:
        # Calculate centrality metrics
        # Use try-except blocks for robustness if graph is small/disconnected
        try:
            pagerank = nx.pagerank(G)
        except nx.NetworkXError:
            pagerank = {node: 0 for node in G.nodes()}

        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        try:
             # Calculate betweenness centrality if graph isn't too large
            if len(G) <= 150:  # Adjusted limit
                betweenness = nx.betweenness_centrality(G)
            else:
                # Use approximate betweenness for larger graphs
                # k samples, max k = num_nodes
                k_samples = min(len(G), max(10, int(len(G) * 0.1)))
                betweenness = nx.betweenness_centrality(G, k=k_samples, normalized=True)
        except Exception: # Catch potential errors in betweenness calculation
             app.logger.warning("Betweenness centrality calculation failed.")
             betweenness = {node: 0 for node in G.nodes()}

        # Combine results
        results = []
        for node in G.nodes():
            results.append({
                'node': node,
                'pagerank': pagerank.get(node, 0),
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'betweenness': betweenness.get(node, 0)
            })

        # Sort by PageRank (most important first)
        results.sort(key=lambda x: x['pagerank'], reverse=True)

        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error calculating centrality: {e}")
        return jsonify({"error": "Failed to calculate centrality metrics."}), 500


@app.route('/api/shortest_path')
def find_shortest_path():
    """Find shortest path between two nodes."""
    source = request.args.get('source')
    target = request.args.get('target')

    if not source or not target:
        return jsonify({'error': 'Source and target parameters are required'}), 400

    # Replace underscores just in case they come from input fields
    source = source.replace('_', ' ')
    target = target.replace('_', ' ')

    if source not in G or target not in G:
        missing = []
        if source not in G: missing.append(f"'{source}'")
        if target not in G: missing.append(f"'{target}'")
        return jsonify({'error': f'Node(s) not found in graph: {", ".join(missing)}'}), 404

    try:
        path = nx.shortest_path(G, source=source, target=target)
        path_data = []

        # Get edge information for the path
        for i in range(len(path) - 1):
            path_data.append({
                'source': path[i],
                'target': path[i + 1]
            })

        return jsonify({
            'path': path,
            'path_data': path_data,
            'length': len(path) - 1
        })
    except nx.NetworkXNoPath:
        return jsonify({'error': f'No path exists between "{source}" and "{target}"'}), 404
    except Exception as e:
        app.logger.error(f"Error finding shortest path between '{source}' and '{target}': {e}")
        return jsonify({"error": "Failed to find shortest path."}), 500


@app.route('/api/communities')
def detect_communities():
    """Detect communities in the graph using Louvain method."""
    if len(G) < 2: # Need at least 2 nodes for community detection
        return jsonify({'communities': [], 'node_communities': []}) # Return empty instead of error

    try:
        # Convert directed graph to undirected for community detection
        # Important: Louvain works best on undirected graphs. Consider if this is appropriate.
        if len(G.edges()) == 0: # Handle graphs with nodes but no edges
             partition = {node: i for i, node in enumerate(G.nodes())}
        else:
            G_undirected = G.to_undirected()
            # Apply Louvain community detection
            partition = community_louvain.best_partition(G_undirected)

        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)

        # Convert to list and sort by community size
        community_list = [{'id': cid, 'nodes': nodes} for cid, nodes in communities.items()]
        community_list.sort(key=lambda x: len(x['nodes']), reverse=True)

        # Add community info to nodes
        node_communities = []
        for node in G.nodes():
            # Node might not be in partition if G_undirected had isolates not in G's edges
            if node in partition:
                node_communities.append({
                    'node': node,
                    'community': partition[node]
                })
            else:
                 node_communities.append({ # Assign isolates their own community
                    'node': node,
                    'community': -1 # Or some other indicator
                })


        return jsonify({
            'communities': community_list,
            'node_communities': node_communities
        })
    except Exception as e:
         app.logger.error(f"Error detecting communities: {e}")
         return jsonify({"error": "Failed to detect communities."}), 500


@app.route('/api/anomalies')
def detect_anomalies():
    """Detect anomalous nodes in the graph based on degree heuristics."""
    if len(G) < 3: # Need a few nodes for meaningful stats
        return jsonify([]) # Return empty list

    try:
        # Calculate statistics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        try:
             pagerank = nx.pagerank(G)
        except nx.NetworkXError:
             pagerank = {node: 0 for node in G.nodes()}


        # Filter out nodes with 0 degree for std calculation if they skew results
        in_degree_values = [d for d in in_degrees.values() if d > 0]
        out_degree_values = [d for d in out_degrees.values() if d > 0]

        # Avoid calculation errors if all degrees are 0 or only one node exists
        in_mean = np.mean(in_degree_values) if in_degree_values else 0
        in_std = np.std(in_degree_values) if len(in_degree_values) > 1 else 0
        out_mean = np.mean(out_degree_values) if out_degree_values else 0
        out_std = np.std(out_degree_values) if len(out_degree_values) > 1 else 0

        # Define thresholds (e.g., 2 standard deviations)
        in_threshold = 2
        out_threshold = 2

        anomalies = []

        for node in G.nodes():
            is_anomaly = False
            anomaly_type = []
            in_d = in_degrees.get(node, 0)
            out_d = out_degrees.get(node, 0)

            # Check in-degree (only if std is meaningful)
            if in_std > 0 and abs(in_d - in_mean) > in_threshold * in_std:
                 is_anomaly = True
                 anomaly_type.append('high_in_degree' if in_d > in_mean else 'low_in_degree')
            elif in_std == 0 and in_mean > 0 and in_d != in_mean: # Handle case where all non-zero degrees are the same
                 is_anomaly = True
                 anomaly_type.append('deviant_in_degree')


            # Check out-degree (only if std is meaningful)
            if out_std > 0 and abs(out_d - out_mean) > out_threshold * out_std:
                is_anomaly = True
                anomaly_type.append('high_out_degree' if out_d > out_mean else 'low_out_degree')
            elif out_std == 0 and out_mean > 0 and out_d != out_mean: # Handle case where all non-zero degrees are the same
                is_anomaly = True
                anomaly_type.append('deviant_out_degree')


            # Orphan pages (no incoming links, but outgoing links exist)
            if in_d == 0 and out_d > 0:
                is_anomaly = True
                anomaly_type.append('orphan')

            # Dead-end pages (no outgoing links, but incoming links exist)
            if out_d == 0 and in_d > 0:
                is_anomaly = True
                anomaly_type.append('dead_end')

            # Isolated nodes (no links in or out) - might be considered anomalies
            if in_d == 0 and out_d == 0 and len(G) > 1:
                 is_anomaly = True
                 anomaly_type.append('isolated')


            if is_anomaly:
                anomalies.append({
                    'node': node,
                    'type': anomaly_type,
                    'in_degree': in_d,
                    'out_degree': out_d,
                    'pagerank': pagerank.get(node, 0)
                })

        return jsonify(anomalies)
    except Exception as e:
        app.logger.error(f"Error detecting anomalies: {e}")
        return jsonify({"error": "Failed to detect anomalies."}), 500


@app.route('/api/predict_links')
def predict_links():
    """Predict potential missing links using common neighbors heuristic."""
    if len(G) < 3 or len(G.edges()) < 2 : # Need some structure
        return jsonify([]) # Return empty list

    predictions = []
    # Convert to undirected for simpler neighbor checks for this heuristic
    # This finds pairs with common neighbors, regardless of original direction.
    # A -> C <- B implies A and B might be related.
    G_undirected = G.to_undirected()

    # Consider Adamic/Adar or Jaccard coefficient for better prediction
    # Simple common neighbors for now:
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            # Skip if already connected (in either direction in original graph)
            if G.has_edge(node1, node2) or G.has_edge(node2, node1):
                continue

            # Get common neighbors in the undirected graph
            try:
                common_neighbors = list(nx.common_neighbors(G_undirected, node1, node2))
                score = len(common_neighbors) # Simple count

                # Add Adamic/Adar score (example)
                # score = sum(1 / np.log(G_undirected.degree(neighbor))
                #             for neighbor in common_neighbors if G_undirected.degree(neighbor) > 1)


                if score >= 1:  # Threshold for prediction (adjust as needed)
                    predictions.append({
                        # Suggest both directions or pick one? Let's suggest A->B and B->A separately if needed
                        # For now, just list the pair
                        'u': node1,
                        'v': node2,
                        'score': score,
                        'common_neighbors': common_neighbors # Optional: send neighbor list
                    })
            except nx.NetworkXError:
                 continue # May happen if nodes have no neighbours


    # Sort by score (most likely first)
    predictions.sort(key=lambda x: x['score'], reverse=True)

    # Return top N predictions
    return jsonify(predictions[:15])


if __name__ == '__main__':
    # Use debug=False for production
    app.run(debug=True, threaded=True) 