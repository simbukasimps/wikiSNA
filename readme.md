# Wikipedia Graph Explorer

This web application allows you to explore Wikipedia by visualizing the connections between articles as a directed graph. The application uses Python (Flask) on the backend for web scraping and graph analysis, and JavaScript (D3.js) on the frontend for visualization.

## Features

- **Interactive Graph Visualization**: Explore Wikipedia articles and see how they are connected.
- **Advanced Graph Analysis**:
  - **Centrality Analysis**: Find the most important articles in the network.
  - **Path Analysis**: Find the shortest path between any two articles.
  - **Community Detection**: Identify clusters of related articles.
  - **Anomaly Detection**: Find articles with unusual connection patterns.
  - **Link Prediction**: Discover potential missing links between articles.

## Installation

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/wikipedia-graph-explorer.git
   cd wikipedia-graph-explorer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the project structure:

   Create the following directories if they don't exist:
   ```
   mkdir -p static/js static/css static/img templates
   ```

5. Place the files in the correct locations:
   - `app-py.py` → root directory
   - `index-html.html` → rename to `index.html` and place in the `templates` folder
   - `style-css.css` → place in `static/css` folder as `style.css`
   - `main.js` → place in `static/js` folder
   - `vec2.js` → place in `static/js` folder

6. Download required images:
   You'll need to download these images for the UI:
   - leftArrow.svg and rightArrow.svg → place in `static/img`
   - GitHub-Mark-32px.png → place in `static/img`

   You can find these images online or create your own.

## Running the Application

1. Make sure your virtual environment is activated.

2. Start the Flask server:
   ```
   python app-py.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

4. To begin exploring:
   - Search for a Wikipedia article in the search box, or
   - Click the "Random Article" button to start with a random article.

5. Click on links in the article or nodes in the graph to explore connections.

6. Use the analysis tabs on the right panel to explore graph properties.

## How It Works

- The backend (Flask) fetches Wikipedia pages and extracts links.
- As you navigate, a graph structure is built and stored using NetworkX.
- Various graph algorithms are applied to analyze the network structure.
- The frontend visualizes the graph using D3.js and displays analysis results.

## Persistence

The graph structure is automatically saved to a file (`wikipedia_graph.pickle`) and will be loaded when you restart the application, so your exploration can continue across sessions.

## Notes

- The graph is rendered in real-time and may slow down with very large graphs.
- Some analyses like betweenness centrality are computationally intensive and may be slow for large graphs.
- For best performance, periodically clear nodes that aren't central to your exploration.
