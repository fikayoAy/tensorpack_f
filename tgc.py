#!/usr/bin/env python3
"""
TensorPack Graph Traversal Module

Implements graph traversal commands for exploring connections between datasets,
finding entity matches, and pathway exploration. This module encapsulates all
functionality specific to the traverse-graph command.
"""

import os
import glob
import time
import logging
import shlex
import numpy as np
import json
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# Import from parent tensorpack module
from matrixtransformer import MatrixTransformer
import script

# Functions used from script module
configure_ocr = script.configure_ocr
setup_logging = script.setup_logging

save_to_file = script.save_to_file
create_parser = script.create_parser

# Entity matching and processing helpers
_OCR_CONFIG = script._OCR_CONFIG
_OCR_INDEX_REGISTRY = script._OCR_INDEX_REGISTRY
_OCR_DATA_REGISTRY = script._OCR_DATA_REGISTRY
_ENTITY_REGISTRY = script._ENTITY_REGISTRY

# ===== Main Command Function =====

def traverse_graph_command(args) -> int:
    """
    Explore semantic pathways and connections between datasets.
    
    This command enables users to navigate and discover relationships between datasets,
    find pathways between specific data points, and explore how different datasets 
    connect semantically without performing any transformations.
    
    Provides standard exports (GraphML, JSON, CSV) for compatibility with external
    network analysis tools like Gephi, Cytoscape, and Neo4j.
    """
    try:
        configure_ocr(extract_on_load=True, video_processing=True)  # Enable OCR and video OCR

        # Handle the case where args is a list (from Python API call)
        if isinstance(args, list):
            parser = create_parser()
            args = parser.parse_args(args)

        # Setup logging first
        setup_logging(args.verbose if hasattr(args, 'verbose') else False,
                      getattr(args, 'log', None))



        start_time = time.time()
        transformer = MatrixTransformer()

        # Expand input file patterns
        input_files = []
        inputs = getattr(args, 'inputs', None)

        # If inputs attribute is missing, check for source/target attributes
        if not inputs and hasattr(args, 'source') and args.source:
            input_files.append(args.source)
            if hasattr(args, 'target') and args.target:
                input_files.append(args.target)
        # Process inputs attribute if available
        elif inputs:
            for pattern in inputs:
                if '*' in pattern or '?' in pattern:
                    input_files.extend(glob.glob(pattern))
                else:
                    input_files.append(pattern)

        if not input_files:
            logging.error("No input files found matching the specified patterns")
            return 1

        print(f"Exploring pathways across {len(input_files)} datasets...")

        # OCR is already enabled in the command handler
        # We can optionally adjust OCR configuration based on specific needs
        if hasattr(args, 'search_entity') and args.search_entity:
            # For entity search, we want to ensure high recall
            configure_ocr(
                confidence_threshold=0.4,  # Lower threshold for better recall in search
                include_in_search=True     # Ensure OCR text is included in search
            )

        # Load datasets for exploration
        datasets = []
        dataset_info = []

        for i, file_path in enumerate(input_files):
            try:
                # Load dataset
                data = script.load_tensor_from_file(file_path)
                datasets.append(data)

                # Log transform application if applied
                if getattr(args, 'apply_transform', None):
                    print(f"  Applied transform '{args.apply_transform}' to {os.path.basename(file_path)}")

                # Extract semantic information directly from data content and structure
                info = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'index': i,
                    'shape': data.shape,
                    'data_type': script._detect_data_type(file_path),
                    'matrix_type': transformer._detect_matrix_type(data),
                    'semantic_coords': transformer._generate_matrix_coordinates(data, i)
                }
                dataset_info.append(info)

                if i % 5 == 0:
                    print(f"Loaded {i+1}/{len(input_files)} datasets...")

            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {str(e)}")
                continue

        if not datasets:
            logging.error("No valid datasets were loaded for exploration")
            return 1

        print(f"Ready to explore {len(datasets)} datasets")

        # Perform pathfinding exploration
        exploration_results = {}

        # Handle different exploration modes
        if getattr(args, 'verbose', False):
            logging.debug(f"Checking exploration mode...")
            logging.debug(f"Has source_dataset: {hasattr(args, 'source_dataset')}")
            logging.debug(f"Has target_dataset: {hasattr(args, 'target_dataset')}")
            logging.debug(f"Has find_bridges: {hasattr(args, 'find_bridges')}")
            logging.debug(f"Has search_entity: {hasattr(args, 'search_entity')}")
            if hasattr(args, 'search_entity'):
                logging.debug(f"search_entity value: {args.search_entity}")

        # Check for source and target, either from source_dataset/target_dataset attributes
        # or from source/target attributes
        source_dataset = None
        target_dataset = None

        if hasattr(args, 'source_dataset') and args.source_dataset:
            source_dataset = args.source_dataset
        elif hasattr(args, 'source') and args.source:
            source_dataset = args.source

        if hasattr(args, 'target_dataset') and args.target_dataset:
            target_dataset = args.target_dataset
        elif hasattr(args, 'target') and args.target:
            target_dataset = args.target

        if source_dataset and target_dataset:
            # Specific path finding between two datasets
            logging.info("Using dataset pathway exploration mode")
            exploration_results = _explore_dataset_pathway(
                transformer, datasets, dataset_info,
                source_dataset, target_dataset, args
            )
            # Visualization removed: log summary instead of rendering rich table
            logging.info("Pathway exploration completed. Visualization suppressed by configuration.")

        elif hasattr(args, 'find_bridges') and args.find_bridges:
            # Find datasets that act as semantic bridges
            logging.info("Using semantic bridges exploration mode")
            exploration_results = _find_semantic_bridges(
                transformer, datasets, dataset_info, args
            )
            # Visualization removed: log summary instead of rendering bridge finder results
            logging.info(f"Bridge-finder exploration completed with {len(exploration_results or [])} results. Visualization suppressed.")

        elif hasattr(args, 'search_entity') and args.search_entity:
            # Search for specific entities across datasets
            # Parse multiple entities if separated by spaces (unless quoted)
            search_entities = []
            
            # Process the search_entity string to handle quoted phrases and multiple terms
            if args.search_entity:
                try:
                    # Use shlex to properly handle quoted phrases
                    search_entities = shlex.split(args.search_entity)
                except ValueError:
                    # Fallback if shlex parsing fails
                    search_entities = [args.search_entity]
            
            if not search_entities:
                search_entities = [args.search_entity]  # Fallback to original string
                
            if len(search_entities) == 1:
                # Single entity search - use existing path
                logging.info(f"Using entity search mode for: {search_entities[0]}")
                exploration_results = _search_entities_across_datasets(
                    transformer, datasets, dataset_info, search_entities[0], args
                )
                logging.info(f"Entity search for '{search_entities[0]}' completed with {len(exploration_results or [])} results.")
            else:
                # Multiple entity search
                logging.info(f"Using multi-entity search mode for: {', '.join(search_entities)}")
                
                # Set up combined results
                combined_results = {
                    'matches': [],
                    'entities': search_entities,
                    'entity_results': {},
                    'datasets_with_all_entities': [],
                    'multi_entity_matches': []
                }
                
                # Determine search mode - default to 'and' logic if not specified
                search_mode = getattr(args, 'search_mode', 'and').lower()
                logging.info(f"Using search mode: {search_mode} (entities: {', '.join(search_entities)})")
                
                # Search for each entity separately
                for entity in search_entities:
                    entity_result = _search_entities_across_datasets(
                        transformer, datasets, dataset_info, entity, args
                    )
                    combined_results['entity_results'][entity] = entity_result
                    
                    # Track which datasets matched each entity
                    if entity_result and 'matches' in entity_result:
                        for match in entity_result['matches']:
                            dataset_path = match.get('dataset_path', '')
                            if dataset_path not in [m.get('dataset_path', '') for m in combined_results['matches']]:
                                # Add to overall matches with entity info
                                match['matched_entities'] = [entity]
                                combined_results['matches'].append(match)
                            else:
                                # Update existing match with this entity
                                for existing_match in combined_results['matches']:
                                    if existing_match.get('dataset_path') == dataset_path:
                                        if 'matched_entities' not in existing_match:
                                            existing_match['matched_entities'] = []
                                        existing_match['matched_entities'].append(entity)
                                        # Increase the relevance score for multiple matches
                                        existing_match['relevance'] = existing_match.get('relevance', 0) + match.get('relevance', 0)
                
                # Filter datasets based on search mode ('and' requires all entities, 'or' accepts any)
                for match in combined_results['matches']:
                    match_entities = set(match.get('matched_entities', []))
                    
                    if search_mode == 'and' and len(match_entities) == len(search_entities):
                        # For AND logic: dataset contains ALL searched entities
                        combined_results['datasets_with_all_entities'].append(match)
                        # Add to multi-entity matches with enhanced scoring
                        enhanced_match = match.copy()
                        enhanced_match['multi_entity_score'] = match.get('relevance', 0) * 1.5  # Boost score for matching all entities
                        combined_results['multi_entity_matches'].append(enhanced_match)
                    elif search_mode == 'or' and len(match_entities) > 0:
                        # For OR logic: dataset contains ANY searched entity
                        # We still track those with all entities separately
                        if len(match_entities) == len(search_entities):
                            combined_results['datasets_with_all_entities'].append(match)
                        
                        # Add to multi-entity matches with score proportional to match count
                        enhanced_match = match.copy()
                        # Score is weighted by the fraction of entities matched
                        match_ratio = len(match_entities) / len(search_entities)
                        enhanced_match['multi_entity_score'] = match.get('relevance', 0) * (1.0 + 0.5 * match_ratio)
                        combined_results['multi_entity_matches'].append(enhanced_match)
                
                # Sort results by relevance or multi-entity score
                combined_results['matches'].sort(key=lambda x: x.get('relevance', 0), reverse=True)
                if combined_results['multi_entity_matches']:
                    combined_results['multi_entity_matches'].sort(key=lambda x: x.get('multi_entity_score', 0), reverse=True)
                
                # Count matches for reporting
                match_count = len(combined_results['matches'])
                all_entities_match_count = len(combined_results['datasets_with_all_entities'])
                
                # Set pathway_type for compatibility with _print_exploration_summary
                combined_results['pathway_type'] = 'multi_entity_search'
                combined_results['search_query'] = ', '.join(search_entities)
                combined_results['matches_found'] = match_count
                combined_results['search_mode'] = search_mode
                
                if search_mode == 'and':
                    logging.info(f"Multi-entity AND search completed with {all_entities_match_count} datasets containing ALL entities.")
                else:
                    logging.info(f"Multi-entity OR search completed with {match_count} total matches. {all_entities_match_count} datasets contain ALL entities.")
                
                # Use the combined results as our exploration results
                exploration_results = combined_results
        else:
            # No specific exploration mode specified - show available options
            print("No exploration mode specified!")
            print("\nAvailable exploration modes:")
            print("  Bridge Discovery:     --find-bridges")
            print("  Entity Search:        --search-entity 'ENTITY_NAME' [--generate-viz]")
            print("  Multiple Entity:      --search-entity \"ENTITY_1 ENTITY_2 ENTITY_3\" [--generate-viz]")
            print("  Quoted Entity:        --search-entity \"'phrase with spaces' another_term\" [--generate-viz]")
            print("  Point-to-Point:       --source-dataset file1.csv --target-dataset file2.csv")
            print("\nExample:")
            print("  python tensorpack.py traverse-graph --inputs *.csv --find-bridges")
            print("  python tensorpack.py traverse-graph --inputs *.csv --search-entity \"diabetes blood_pressure\"")
            return

        # Export exploration results in multiple formats
        output_base = getattr(args, 'output', 'pathway_results')

        # Create directory for output if it doesn't exist
        output_dir = os.path.dirname(output_base)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure shared_entities have numeric confidence before exporting
        try:
            if isinstance(exploration_results, dict) and 'shared_entities' in exploration_results:
                for ent in exploration_results.get('shared_entities', []):
                    try:
                        if ent is None:
                            continue
                        if 'confidence' not in ent or ent.get('confidence') is None:
                            ent['confidence'] = script._compute_shared_entity_confidence(ent)
                    except Exception:
                        # Defensive: don't let a single bad entity stop exporting
                        pass
        except Exception:
            pass

        # JSON export (always do this as the primary format)
        json_path = f"{output_base}.json"
        save_to_file(exploration_results, json_path, 'json')
        print(f"Results saved to: {json_path}")

        # === NEW: Save main numeric results as .npy/.npz if requested ===
        numeric_arrays = {}
        # Find main numeric arrays in exploration_results (top-level and one level nested)
        if isinstance(exploration_results, dict):
            for k, v in exploration_results.items():
                if isinstance(v, (np.ndarray, list)):
                    arr = np.array(v)
                    if arr.dtype.kind in 'fiubc' and arr.size > 0:
                        numeric_arrays[k] = arr
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (np.ndarray, list)):
                            arr = np.array(vv)
                            if arr.dtype.kind in 'fiubc' and arr.size > 0:
                                numeric_arrays[f"{k}_{kk}"] = arr

        if getattr(args, 'save_npy', False):
            for name, arr in numeric_arrays.items():
                npy_path = f"{output_base}_{name}.npy"
                np.save(npy_path, arr)
                print(f"Saved numeric result: {npy_path}")

        if getattr(args, 'save_npz', False):
            npz_path = f"{output_base}.npz"
            np.savez(npz_path, **numeric_arrays)
            print(f"Saved all numeric results in: {npz_path}")

        # convertto support intentionally removed: additional export formats are not available
        # All export generation beyond the primary JSON is skipped in this build.

        # Print completion summary
        elapsed_time = time.time() - start_time
        print(f"\nExploration completed in {elapsed_time:.2f} seconds")
        print(f"Processed {len(datasets)} datasets")

        return 0
    except Exception as e:
        logging.error(f"Error during dataset exploration: {str(e)}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1

# ===== Helper functions for recursive data structure handling =====

def _recursively_remove_local_relationships(data):
    """
    Recursively remove local_relationships from nested data structures.
    This reduces the size of exported results by removing redundant information.
    """
    if isinstance(data, dict):
        if 'local_relationships' in data:
            del data['local_relationships']
        for key, value in data.items():
            _recursively_remove_local_relationships(value)
    elif isinstance(data, list):
        for item in data:
            _recursively_remove_local_relationships(item)
    return data

# ===== Pathway exploration functions =====

def _explore_dataset_pathway(transformer, datasets, dataset_info, source_dataset, target_dataset, args):
    """
    Find pathways between specific source and target datasets.
    
    Explores the semantic connections and potential transformation pathways
    that could be applied to reach from one dataset to another.
    """
    # Find source and target datasets by name or index
    source_idx = _find_dataset_index(datasets, dataset_info, source_dataset)
    target_idx = _find_dataset_index(datasets, dataset_info, target_dataset)
    
    if source_idx is None:
        logging.error(f"Source dataset '{source_dataset}' not found")
        return {'error': f"Source dataset '{source_dataset}' not found"}
    
    if target_idx is None:
        logging.error(f"Target dataset '{target_dataset}' not found")
        return {'error': f"Target dataset '{target_dataset}' not found"}
    
    # Get the actual datasets
    source_data = datasets[source_idx]
    target_data = datasets[target_idx]
    source_info = dataset_info[source_idx]
    target_info = dataset_info[target_idx]
    
    logging.info(f"Finding pathway from {source_info['file_name']} to {target_info['file_name']}...")
    
    # Find direct semantic connections between the datasets
    semantic_connections = script._find_semantic_connections_between_datasets(
        source_data, target_data, source_info, target_info
    )
    
    # Find contextual bridges between the datasets
    contextual_bridges = _find_contextual_bridges(
        source_info.get('metadata', {}), target_info.get('metadata', {})
    )
    
    # Get any shared entities between the datasets
    entity_overlaps = script._extract_key_entities_from_dataset(source_data, source_info)
    
    # Calculate a confidence score for the connection
    connection_strength = script._calculate_semantic_distance(
        source_info['semantic_coords'],
        target_info['semantic_coords']
    )
    
    # Get potential transformations that could be applied
    transformation_suggestions = []
    # Transformation analysis would go here if needed
    
    # Assess the quality of the connection
    connection_quality = {
        'overall_quality': float(1.0 - connection_strength),
        'has_semantic_connections': len(semantic_connections) > 0,
        'has_contextual_bridges': len(contextual_bridges) > 0,
        'has_entity_overlap': len(entity_overlaps) > 0
    }
    
    # Connection meaning and data flow analysis
    connection_meaning = f"Semantic distance: {connection_strength:.3f}"
    data_flow = {'bidirectional': True}
    
    # Build the result dictionary
    result = {
        'pathway_type': 'direct_exploration',
        'source_dataset': source_info,
        'target_dataset': target_info,
        'connection_strength': float(connection_strength),
        'connection_quality': connection_quality,
        'semantic_connections': semantic_connections,
        'contextual_bridges': contextual_bridges,
        'transformation_suggestions': transformation_suggestions,
        'shared_entities': entity_overlaps,
        'connection_meaning': connection_meaning,
        'data_flow_potential': data_flow,
        'direct_connection': connection_strength < 0.7  # Threshold for direct connection
    }
    
    # For large result sets, remove verbose local relationships to reduce size
    return _recursively_remove_local_relationships(result)

def _find_dataset_index(datasets, dataset_info, identifier):
    """Find the index of a dataset by name, path, or direct index."""
    if identifier is None:
        return None
    
    # Try to parse as integer index
    try:
        idx = int(identifier)
        if 0 <= idx < len(datasets):
            return idx
    except (ValueError, TypeError):
        pass
    
    # Try to find by filename or path
    for i, info in enumerate(dataset_info):
        if (identifier == info['file_name'] or 
            identifier == info['file_path'] or
            info['file_path'].endswith(identifier)):
            return i
    
    return None

def _find_contextual_bridges(source_metadata, target_metadata):
    """
    Find concepts, entities, or themes that connect two datasets via their metadata.
    
    Examines the metadata for shared concepts that might indicate a semantic bridge
    between otherwise different datasets.
    """
    if not source_metadata or not target_metadata:
        return []
    
    bridges = []
    
    # Check for shared keys in metadata (direct bridges)
    shared_keys = set(source_metadata.keys()) & set(target_metadata.keys())
    for key in shared_keys:
        # Skip standard metadata fields
        if key in ('shape', 'dtype', 'file_path', 'file_name', 'timestamp'):
            continue
            
        # Examine values for similarity
        source_val = source_metadata[key]
        target_val = target_metadata[key]
        
        # Convert values to strings for comparison
        if not isinstance(source_val, str):
            source_val = str(source_val)
        if not isinstance(target_val, str):
            target_val = str(target_val)
        
        similarity = script._calculate_text_similarity(source_val, target_val)
        
        if similarity > 0.5:  # Only add if reasonably similar
            bridges.append({
                'bridge_type': 'shared_metadata_field',
                'field_name': key,
                'similarity': similarity,
                'source_value': source_metadata[key],
                'target_value': target_metadata[key]
            })
    
    # Check for conceptual similarity in descriptions
    if ('description' in source_metadata and 'description' in target_metadata):
        desc_similarity = script._calculate_text_similarity(
            source_metadata['description'], target_metadata['description']
        )
        
        if desc_similarity > 0.3:  # Lower threshold for descriptions
            bridges.append({
                'bridge_type': 'conceptual_description',
                'similarity': desc_similarity,
                'summary': 'Conceptual similarity in dataset descriptions'
            })
    
    # Add any entity crossovers if available
    if ('entities' in source_metadata and 'entities' in target_metadata):
        source_entities = source_metadata['entities']
        target_entities = target_metadata['entities']
        
        if isinstance(source_entities, list) and isinstance(target_entities, list):
            # Find common entity names
            source_entity_names = {e.get('name', '').lower() for e in source_entities if isinstance(e, dict)}
            target_entity_names = {e.get('name', '').lower() for e in target_entities if isinstance(e, dict)}
            
            common_entities = source_entity_names & target_entity_names
            if common_entities:
                bridges.append({
                    'bridge_type': 'shared_entities',
                    'count': len(common_entities),
                    'entities': list(common_entities)
                })
    
    return bridges

def _find_semantic_bridges(transformer, datasets, dataset_info, args):
    """
    Find datasets that act as semantic bridges between other datasets.
    
    Identifies datasets that have connections to multiple other datasets,
    potentially allowing them to serve as connectors between otherwise
    disconnected data points.
    """
    n_datasets = len(datasets)
    if n_datasets < 3:
        logging.warning("Need at least 3 datasets to find meaningful bridges")
        return {'bridge_datasets': [], 'error': 'Not enough datasets for bridge analysis'}
    
    logging.info(f"Analyzing {n_datasets} datasets for semantic bridges...")
    
    # Calculate pairwise connections between all datasets
    connection_matrix = np.zeros((n_datasets, n_datasets))
    connection_details = {}
    
    # Establish a minimum threshold for connections to be considered
    min_connection_threshold = 0.3
    
    # Number of pairs to process
    total_pairs = (n_datasets * (n_datasets - 1)) // 2
    processed_pairs = 0
    
    # Track bridges for each dataset
    dataset_connections = [[] for _ in range(n_datasets)]
    connection_types = [set() for _ in range(n_datasets)]
    
    # Process each unique pair of datasets
    for i in range(n_datasets):
        for j in range(i+1, n_datasets):
            if processed_pairs % 10 == 0:
                logging.info(f"Processed {processed_pairs}/{total_pairs} dataset pairs...")
            
            # Find connections between these two datasets
            source_data = datasets[i]
            target_data = datasets[j]
            source_info = dataset_info[i]
            target_info = dataset_info[j]
            
            # Compute semantic distance (lower is better/closer)
            dist = script._calculate_semantic_distance(
                source_info['semantic_coords'],
                target_info['semantic_coords']
            )
            
            # Convert to connection strength (higher is better/stronger)
            strength = 1.0 - min(dist, 1.0)
            
            if strength > min_connection_threshold:
                connection_matrix[i, j] = connection_matrix[j, i] = strength
                
                # If strength is good enough, dig deeper for detailed connection info
                if strength > 0.4:
                    semantic_connections = script._find_semantic_connections_between_datasets(
                        source_data, target_data, source_info, target_info
                    )
                    
                    contextual_bridges = _find_contextual_bridges(
                        source_info.get('metadata', {}), 
                        target_info.get('metadata', {})
                    )
                    
                    # If we found significant connections, record them
                    if semantic_connections or contextual_bridges:
                        connection_details[(i, j)] = {
                            'strength': strength,
                            'semantic_connections': semantic_connections,
                            'contextual_bridges': contextual_bridges
                        }
                        
                        # Track connection for each dataset
                        dataset_connections[i].append((j, strength))
                        dataset_connections[j].append((i, strength))
                        
                        # Track connection types
                        if semantic_connections:
                            connection_types[i].add('semantic')
                            connection_types[j].add('semantic')
                        if contextual_bridges:
                            connection_types[i].add('contextual')
                            connection_types[j].add('contextual')
            
            processed_pairs += 1
    
    logging.info(f"Finished analyzing {processed_pairs} dataset pairs")
    
    # Find datasets with the most connections
    connection_counts = [len(conns) for conns in dataset_connections]
    
    # Find potential bridge datasets (those with connections to multiple others)
    potential_bridges = []
    for i in range(n_datasets):
        connections = dataset_connections[i]
        if len(connections) >= 2:  # Need at least 2 connections to be a bridge
            # Calculate average connection strength
            avg_strength = sum(s for _, s in connections) / len(connections)
            
            # Calculate bridge score (combination of connection count and strength)
            bridge_score = (len(connections) * 0.6) + (avg_strength * 0.4)
            
            # Get unique data types this bridges between
            connected_data_types = set()
            for j, _ in connections:
                connected_data_types.add(dataset_info[j]['data_type'])
                
            # Higher score if it bridges different data types
            if len(connected_data_types) > 1:
                bridge_score *= 1.2
            
            # Higher score if it has multiple connection types
            if len(connection_types[i]) > 1:
                bridge_score *= 1.1
                
            potential_bridges.append({
                'bridge_dataset': dataset_info[i],
                'bridge_score': bridge_score,
                'connectivity_count': len(connections),
                'avg_connection_strength': avg_strength,
                'connected_data_types': list(connected_data_types),
                'connection_types': list(connection_types[i])
            })
    
    # Sort by bridge score (higher is better)
    potential_bridges.sort(key=lambda x: x['bridge_score'], reverse=True)
    
    # Generate semantic interpretations for all datasets (matching dcc.py structure)
    interpreted_coordinates = {}
    for idx, info in enumerate(dataset_info):
        interpreted_coordinates[idx] = script._interpret_semantic_coordinates(
            info['semantic_coords'], info, {}
        )
    
    # Prepare comprehensive dataset info (matching dcc.py output structure)
    comprehensive_dataset_info = {}
    for idx, info in enumerate(dataset_info):
        # Convert semantic_coords to list if it's a numpy array
        semantic_coords = info['semantic_coords']
        if isinstance(semantic_coords, np.ndarray):
            semantic_coords = semantic_coords.tolist()
        elif not isinstance(semantic_coords, list):
            semantic_coords = list(semantic_coords) if hasattr(semantic_coords, '__iter__') else [semantic_coords]
        
        comprehensive_dataset_info[idx] = {
            'file_info': {
                'file_path': info['file_path'],
                'file_name': info['file_name'],
                'index': idx,
                'shape': list(info['shape']) if hasattr(info['shape'], '__iter__') else info['shape'],
                'data_type': info['data_type'],
                'matrix_type': info['matrix_type'],
                'semantic_coords': semantic_coords
            },
            'semantic_interpretation': interpreted_coordinates[idx],
            'matrix_analysis': {
                'matrix_type': info['matrix_type'],
                'data_type': info['data_type'],
                'shape': list(info['shape']) if hasattr(info['shape'], '__iter__') else info['shape'],
                'complexity_estimate': script.get_complexity(info['matrix_type'], info['shape']),
                'memory_efficiency': script.get_memory_efficiency(info['matrix_type'])
            }
        }
    
    # Prepare the final result with comprehensive information
    result = {
        'pathway_type': 'semantic_bridges',
        'bridge_datasets': potential_bridges,
        'total_datasets': n_datasets,
        'connection_matrix': connection_matrix.tolist(),
        'dataset_info': comprehensive_dataset_info
    }
    
    return result

# ===== Entity Search functions =====

def _search_entities_across_datasets(transformer, datasets, dataset_info, search_entity, args):
    """
    Search for specific entities across all datasets.
    
    Enables finding references to specific entities (like people, organizations, concepts)
    across diverse datasets.
    """
    n_datasets = len(datasets)
    logging.info(f"Searching for '{search_entity}' across {n_datasets} datasets...")
    
    search_results = []
    
    # Search for the entity in each dataset
    for i, dataset in enumerate(datasets):
        info = dataset_info[i]
        
        # Implement entity matching strategy pattern
        matches = script._find_entity_matches_in_dataset(dataset, search_entity, info)
        
        if matches:
            result = {
                'dataset_path': info['file_path'],
                'dataset_name': info['file_name'],
                'dataset_index': i,
                'data_type': info['data_type'],
                'entity_matches': matches,
                'match_count': len(matches),
                'relevance': sum(m.get('confidence', 0) for m in matches) / len(matches)
            }
            search_results.append(result)
    
    # Sort results by relevance (higher first)
    search_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
    
    # Extract contextual relationships from matches
    for result in search_results:
        dataset_idx = result['dataset_index']
        dataset = datasets[dataset_idx]
        info = dataset_info[dataset_idx]
        
        # Find local contextual relationships
        contextual_info = script._extract_contextual_info(info.get('metadata', {}), search_entity)
        
        # Find local relationships within this dataset
        local_relationships = script._find_local_contextual_relationships(
            dataset, search_entity, info.get('metadata', {}), info
        )
        
        # Add to result
        result['contextual_info'] = contextual_info
        result['local_relationships'] = local_relationships
    
    # Generate entity-centric view of the results
    entity_view = {
        'search_entity': search_entity,
        'total_matches': sum(r['match_count'] for r in search_results),
        'dataset_count': len(search_results),
        'highest_relevance': max([r.get('relevance', 0) for r in search_results], default=0),
        'data_types': list(set(r['data_type'] for r in search_results))
    }
    
    # Prepare the final result
    result = {
        'pathway_type': 'entity_search',
        'search_query': search_entity,
        'matches_found': len(search_results),
        'matches': search_results,
        'entity_view': entity_view
    }
    
    return result

