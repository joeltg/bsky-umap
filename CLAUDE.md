# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses a Makefile-based build system with environment variables. The build system requires a `.env` file with the following variables:
- `DATA`: Path to data directory
- `DIM`: Embedding dimension
- `N_NEIGHBORS`: Number of neighbors for KNN
- `N_CLUSTERS`: Number of clusters for labeling
- `N_EPOCHS`: Number of training epochs
- `N_THREADS`: Number of threads for parallel processing

Common build commands:
```bash
make init          # Initialize data structures from graph.sqlite
make embeddings    # Generate high-dimensional embeddings
make colors        # Generate color mappings for visualization
make umap          # Run UMAP dimensionality reduction
make save          # Save positions and create atlas
make clean         # Clean generated files
```

## Architecture

This is a UMAP-based graph visualization system for Bluesky social network data. The pipeline consists of:

1. **Data Pipeline (Python)**:
   - `sqlite_to_arrow.py`: Converts SQLite graph data to Arrow format
   - `embedding.py`: Generates high-dimensional node embeddings using GGVec
   - `knn.py`: Computes k-nearest neighbors for UMAP
   - `project.py`: Runs UMAP dimensionality reduction
   - `save_graph.py`: Saves 2D positions to SQLite
   - `anneal.py`: Performs force-directed layout optimization
   - `colors.py`: Generates color assignments based on clustering
   - `labels.py`: Performs clustering on embeddings

2. **Tile Generation (Zig)**:
   - `atlas/`: Zig codebase for generating hierarchical tiles
   - `atlas/src/main.zig`: CLI tool for partitioning nodes into quadtree tiles
   - `atlas/src/Atlas.zig`: Spatial indexing data structure for fast nearest neighbor queries
   - Uses quadtree spatial partitioning with configurable capacity per tile

3. **Dependencies**:
   - Python: numpy, scipy, scikit-learn, umap-learn, pyarrow, csrgraph, nodevectors
   - Zig: Custom quadtree implementation for spatial indexing

## Data Flow

The typical processing pipeline:
1. Start with `graph.sqlite` containing node and edge data
2. Convert to Arrow format with `make init`
3. Generate embeddings with `make embeddings`
4. Run UMAP projection with `make umap`
5. Generate tiles with the Zig atlas tool
6. Create color mappings with `make colors`

The atlas tool generates hierarchical tiles for efficient visualization at different zoom levels, with each tile containing position data and color information for nodes in that spatial region.

## Development

Build the Zig atlas tool:
```bash
cd atlas && zig build
```

The project uses environment variables extensively - ensure `.env` is configured before running any commands.