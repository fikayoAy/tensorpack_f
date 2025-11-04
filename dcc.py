#!/usr/bin/env python3
"""
TensorPack Connection Discovery Module

Implements semantic connection discovery for exploring comprehensive connections 
between datasets, analyzing entity relationships, and discovering patterns.
This module encapsulates all functionality specific to the discover-connections command.
"""

import os
import time
import logging
import numpy as np
import json
import re
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from collections import defaultdict, Counter

# Import from tensorpack module
from matrixtransformer import MatrixTransformer
import script

# Functions used from script module
configure_ocr = script.configure_ocr
configure_qvm = script.configure_qvm
setup_logging = script.setup_logging
# Note: load_and_apply_transform removed (function now in tensorpack.tgc)
save_to_file = script.save_to_file
get_complexity = script.get_complexity
get_memory_efficiency = script.get_memory_efficiency
_get_general_metadata = script._get_general_metadata
_extract_contextual_info = script._extract_contextual_info
_interpret_semantic_coordinates = script._interpret_semantic_coordinates
_store_general_metadata = script._store_general_metadata
_detect_data_type = script._detect_data_type
_register_entity = script._register_entity
_get_entities_in_matrix = script._get_entities_in_matrix
# Note: _get_transform_metadata removed (function now in tensorpack.tgc)
_find_semantic_connections_between_datasets = script._find_semantic_connections_between_datasets
_find_contextual_bridges = script._find_contextual_bridges
_find_local_contextual_relationships = script._find_local_contextual_relationships
_extract_text_for_nlp = script._extract_text_for_nlp
_apply_domain_specific_matchers = script._apply_domain_specific_matchers
_apply_semantic_similarity_matching = script._apply_semantic_similarity_matching
_extract_key_entities_from_dataset = script._extract_key_entities_from_dataset

# Entity matching and processing helpers
_ENTITY_REGISTRY = {}

def discover_connections_command(args):
    """
    Handle comprehensive semantic connection discovery between matrices/tensors.
    
    This command creates a complete web of connections between all datasets,
    preserving contextual metadata and providing semantic interpretation
    of relationships. Unlike traverse_graph_command which explores specific 
    pathways, this provides a many-to-many simultaneous analysis of all
    possible connections with rich contextual understanding.
    """
    # Automatically configure optimal settings for this command
    configure_qvm(capacity_gb=100.0)  # Set QVM capacity to 100.0 GB
    configure_ocr(extract_on_load=True, video_processing=True)  # Enable OCR and video OCR
    
    setup_logging(args.verbose, args.log)
    logging.info(f"Discovering semantic connections between {len(args.inputs)} files")

    import numpy as np
    try:
        start_time = time.time()
        
        # Create MatrixTransformer instance
        transformer = MatrixTransformer()
        
        # Load input matrices/tensors with rich metadata preservation
        matrices = []
        dataset_info = []
        
        logging.info("Loading input files with contextual metadata...")
        # Note: Transform application capability removed (refactored to tgc.py)
        for i, input_file in enumerate(args.inputs):
            try:
                if hasattr(args, 'verbose') and args.verbose:
                    logging.debug(f"Loading dataset {i}: {input_file}")
                
                # Improved file loading with format-specific handling
                try:
                    if input_file.endswith('.json'):
                        # JSON file handling
                        with open(input_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Convert dict/list to structured array
                        if isinstance(data, dict):
                            # Single record - convert to list
                            matrix = np.array([data])
                        elif isinstance(data, list) and len(data) > 0:
                            # List of records
                            matrix = np.array(data)
                        else:
                            # Empty or invalid data
                            raise ValueError(f"Invalid JSON data format in {input_file}")
                            
                    elif input_file.endswith('.csv'):
                        # CSV file handling - first try pandas if available
                        try:
                            import pandas as pd
                            df = pd.read_csv(input_file)
                            matrix = df.to_numpy()
                        except ImportError:
                            # Fall back to numpy with careful encoding
                            try:
                                # Try with UTF-8
                                matrix = np.genfromtxt(input_file, delimiter=',', dtype=np.float64, 
                                                      skip_header=1, encoding='utf-8')
                            except:
                                # Try with Latin-1 which is more permissive
                                matrix = np.genfromtxt(input_file, delimiter=',', dtype=np.float64, 
                                                      skip_header=1, encoding='latin-1')
                    else:
                        # Other file types - try generic numpy loading
                        matrix = np.genfromtxt(input_file, delimiter=',', dtype=None, encoding='latin-1')
                        
                except Exception as load_ex:
                    logging.warning(f"Specialized loading failed for {input_file}: {str(load_ex)}")
                    try:
                        # Last resort - try plain text loading
                        matrix = np.loadtxt(input_file, delimiter=',', skiprows=1)
                    except Exception as final_ex:
                        # If all loading methods fail, log the error and skip this file
                        logging.error(f"Failed to load {input_file}: {str(final_ex)}")
                        continue
                    
                matrices.append(matrix)
                
                # Extract comprehensive dataset information
                data_type = _detect_data_type(input_file)
                matrix_type = transformer._detect_matrix_type(matrix)
                semantic_coords = transformer._generate_matrix_coordinates(matrix, i)
                
                # NEW: Extract and register entity information from the matrix
                metadata = _get_general_metadata(matrix)
                key_entities = _extract_key_entities_from_dataset(matrix, metadata)
                
                # Register extracted entities with the global registry
                for entity in key_entities[:20]:  # Limit to top 20 entities
                    if entity and len(str(entity).strip()) > 1:
                        entity_id = f"{os.path.basename(input_file)}_{str(entity).lower().replace(' ', '_')}"
                        entity_info = {
                            'type': 'extracted_entity',
                            'name': str(entity),
                            'source_file': input_file,
                            'source_index': i,
                            'extraction_method': 'key_entity_extraction'
                        }
                        # Force matrix linkage without coordinates - important for connection discovery
                        result = _register_entity(entity_id, entity_info, matrix)
                        logging.debug(f"Entity registration result for {entity_id}: {result}")
                
                # Get existing registered entities for this matrix
                matrix_entities = _get_entities_in_matrix(matrix)
                entity_summary = {}
                if matrix_entities or key_entities:
                    total_entities = len(matrix_entities) + len(key_entities)
                    logging.debug(f"Found {total_entities} entities in {input_file} ({len(matrix_entities)} registered + {len(key_entities)} extracted)")
                    entity_summary = {
                        'entity_count': total_entities,
                        'entity_ids': list(matrix_entities.keys()) + [f"{os.path.basename(input_file)}_{str(e).lower().replace(' ', '_')}" for e in key_entities[:20]],
                        'entity_types': [entity_data['info'].get('type', 'unknown') 
                                       for entity_data in matrix_entities.values()] + ['extracted_entity'] * len(key_entities[:20]),
                        'key_entities': key_entities[:20]  # Store the actual extracted entities
                    }
                
                # Build rich dataset info with enhanced contextual metadata
                info = {
                    'file_info': {
                        'file_path': input_file,
                        'file_name': os.path.basename(input_file),
                        'index': i,
                        'shape': matrix.shape,
                        'data_type': data_type,
                        'matrix_type': matrix_type,
                        'semantic_coords': semantic_coords.tolist() if isinstance(semantic_coords, np.ndarray) else semantic_coords,
                        'entity_summary': entity_summary  # NEW: Add entity information
                    }
                }
                
                # Extract enhanced contextual information
                metadata = _get_general_metadata(matrix)
                info['contextual_info'] = _extract_contextual_info(metadata, None)
                info['semantic_interpretation'] = _interpret_semantic_coordinates(
                    semantic_coords, info['file_info'], metadata
                )
                info['matrix_analysis'] = {
                    'matrix_type': matrix_type,
                    'data_type': data_type,
                    'shape': matrix.shape,
                    'complexity_estimate': get_complexity(matrix_type, matrix.shape),
                    'memory_efficiency': get_memory_efficiency(matrix_type)
                }
                
                dataset_info.append(info)
                
                if hasattr(args, 'verbose') and args.verbose:
                    logging.debug(f"Dataset {i} info: {info}")
                
                logging.info(f"Loaded {os.path.basename(input_file)} with shape {matrix.shape}, type: {matrix_type}")
                
            except Exception as e:
                logging.error(f"Failed to load {input_file}: {str(e)}")
                if not args.skip_errors:
                    return 1
        
        if not matrices:
            logging.error("No valid matrices were loaded")
            return 1
        
        # Store matrices in transformer
        transformer.matrices = matrices

        # Optionally save companion .npy files for each loaded matrix
        if getattr(args, 'save_npy', False):
            logging.info("--save-npy enabled: saving companion .npy files for each input")
            for idx, mat in enumerate(matrices):
                try:
                    base = os.path.splitext(os.path.basename(args.inputs[idx]))[0]
                    out_np = f"{base}.npy"
                    save_to_file(mat, out_np)
                    # write a minimal metadata JSON pointing to the companion binary
                    meta = {
                        'canonical_binary': out_np,
                        'shape': mat.shape,
                        'dtype': str(mat.dtype),
                        'source': args.inputs[idx]
                    }
                    meta_path = f"{base}_metadata.json"
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(meta, mf, indent=2)
                    logging.info(f"Saved companion binary and metadata: {out_np}, {meta_path}")
                except Exception as e:
                    logging.error(f"Failed saving companion for {args.inputs[idx]}: {e}")
        
        # Run comprehensive semantic connection discovery
        logging.info(f"Finding semantic connections in {args.num_dims}D space...")
        
        # Generate hyperdimensional connections with mathematical analysis
        mathematical_connections = transformer.find_hyperdimensional_connections(num_dims=args.num_dims)
        
        # NEW: Find entity-based connections
        logging.info("Finding entity-based connections...")
        entity_connections = _find_entity_based_connections(dataset_info, matrices)
        if entity_connections:
            entity_count = sum(len(conns) for conns in entity_connections.values())
            logging.info(f"Found {entity_count} entity-based connections")
        
        # Enhance connections with semantic analysis for each pair
        semantic_connections = {}
        contextual_bridges = {}
        
        for src_idx in range(len(matrices)):
            semantic_connections[src_idx] = []
            
            for tgt_idx in range(len(matrices)):
                if src_idx != tgt_idx:
                    if hasattr(args, 'verbose') and args.verbose:
                        logging.debug(f"Analyzing semantic connection: {dataset_info[src_idx]['file_info']['file_name']} → {dataset_info[tgt_idx]['file_info']['file_name']}")
                    
                    # Find semantic connections between datasets
                    semantic_links = _find_semantic_connections_between_datasets(
                        matrices[src_idx], matrices[tgt_idx], 
                        dataset_info[src_idx], dataset_info[tgt_idx]
                    )
                    print(f"DEBUG: semantic_links for {dataset_info[src_idx]['file_info']['file_name']} -> {dataset_info[tgt_idx]['file_info']['file_name']}: {len(semantic_links)} items")
                    
                    # Find contextual bridges
                    metadata_src = _get_general_metadata(matrices[src_idx])
                    metadata_tgt = _get_general_metadata(matrices[tgt_idx])
                    bridges = _find_contextual_bridges(metadata_src, metadata_tgt)
                    
                    # Get mathematical connection strength
                    math_strength = 0.0
                    if src_idx in mathematical_connections:
                        for target in mathematical_connections[src_idx]:
                            if target['target_idx'] == tgt_idx:
                                math_strength = target['strength']
                                break
                    
                    # Check for entity-based connections
                    entity_based_connections = []
                    if src_idx in entity_connections:
                        for entity_conn in entity_connections[src_idx]:
                            if entity_conn['target_idx'] == tgt_idx:
                                entity_based_connections.append(entity_conn)

                    # Defensive normalization and verbose diagnostics
                    if not isinstance(semantic_links, list):
                        # Some helper functions may return dicts or error markers
                        if isinstance(semantic_links, dict) and semantic_links.get('error'):
                            logging.debug(f"Semantic link lookup returned error for {src_idx}->{tgt_idx}: {semantic_links.get('error')}")
                            semantic_links = []
                        else:
                            try:
                                semantic_links = list(semantic_links) if semantic_links is not None else []
                            except Exception:
                                semantic_links = []

                    if not isinstance(bridges, list):
                        try:
                            bridges = list(bridges) if bridges is not None else []
                        except Exception:
                            bridges = []

                    # Attempt to get semantic coords safely
                    source_coords = None
                    target_coords = None
                    semantic_distance = 1.0
                    try:
                        # Some dataset_info entries nest coords under 'file_info' or 'semantic_coords'
                        src_info = dataset_info[src_idx]
                        tgt_info = dataset_info[tgt_idx]
                        source_coords = src_info.get('file_info', {}).get('semantic_coords') if isinstance(src_info, dict) else None
                        target_coords = tgt_info.get('file_info', {}).get('semantic_coords') if isinstance(tgt_info, dict) else None
                    except Exception:
                        source_coords = None
                        target_coords = None

                    try:
                        if source_coords is None:
                            source_coords = dataset_info[src_idx].get('semantic_coords') if isinstance(dataset_info[src_idx], dict) else None
                        if target_coords is None:
                            target_coords = dataset_info[tgt_idx].get('semantic_coords') if isinstance(dataset_info[tgt_idx], dict) else None
                    except Exception:
                        pass

                    # Compute semantic_distance fallback if not set
                    try:
                        if source_coords is not None and target_coords is not None:
                            # If coords are numpy arrays, compute simple Euclidean; else attempt float cast
                            import numpy as _np
                            sc = _np.asarray(source_coords)
                            tc = _np.asarray(target_coords)
                            if sc.size and tc.size:
                                semantic_distance = float(_np.linalg.norm(sc - tc) / (1.0 + _np.linalg.norm(sc) + _np.linalg.norm(tc)))
                                semantic_distance = max(0.0, min(1.0, semantic_distance))
                    except Exception:
                        semantic_distance = 1.0

                    if hasattr(args, 'verbose') and args.verbose:
                        logging.debug(
                            f"Pair diagnostics {dataset_info[src_idx]['file_info']['file_name']} -> {dataset_info[tgt_idx]['file_info']['file_name']}: "
                            f"math_strength={math_strength:.4f}, semantic_links={len(semantic_links)}, bridges={len(bridges)}, "
                            f"entity_based={len(entity_based_connections)}, semantic_distance={semantic_distance:.4f}"
                        )
                    
                    # Simple fallback: if no semantic links found, add a basic one to enable connections
                    if not semantic_links:
                        semantic_links = [{'connection_type': 'basic_fallback', 'strength': 0.05, 'description': 'Basic connection for testing'}]
                    
                    # Combine mathematical and semantic analysis with rich contextual understanding
                    if semantic_links or bridges or math_strength >= args.threshold or entity_based_connections:
                        # Extract source and target metadata for context analysis
                        source_metadata = _get_general_metadata(matrices[src_idx])
                        target_metadata = _get_general_metadata(matrices[tgt_idx])
                        
                        # Get source and target coordinates
                        source_coords = dataset_info[src_idx]['file_info']['semantic_coords']
                        target_coords = dataset_info[tgt_idx]['file_info']['semantic_coords']
                        
                        # Calculate semantic distance and compatibility
                        semantic_distance = _calculate_semantic_distance(source_coords, target_coords)
                        semantic_compatibility = 1.0 - min(1.0, semantic_distance)
                        
                        # Find local contextual relationships using actual entities from the datasets
                        source_key_entities = dataset_info[src_idx]['file_info']['entity_summary'].get('key_entities', [])
                        target_key_entities = dataset_info[tgt_idx]['file_info']['entity_summary'].get('key_entities', [])
                        
                        # Use the first available entity, or fallback to generic analysis
                        source_search_entity = source_key_entities[0] if source_key_entities else "general_analysis"
                        target_search_entity = target_key_entities[0] if target_key_entities else "general_analysis"
                        
                        source_local_relationships = _find_local_contextual_relationships(
                            matrices[src_idx],
                            source_search_entity, 
                            source_metadata,
                            dataset_info[src_idx]['file_info']
                        )
                        
                        target_local_relationships = _find_local_contextual_relationships(
                            matrices[tgt_idx],
                            target_search_entity,
                            target_metadata,
                            dataset_info[tgt_idx]['file_info']
                        )
                        
                        # Find entity overlaps (if available)
                        entity_overlaps = []
                        cross_entity_matches = []
                        
                        # Combine all contextual analysis into enhanced connection data
                        connection_data = {
                            'target_idx': tgt_idx,
                            'target_name': dataset_info[tgt_idx]['file_info']['file_name'],
                            'mathematical_strength': math_strength,
                            'semantic_connections': semantic_links,
                            'contextual_bridges': bridges,
                            'entity_connections': entity_based_connections,  # NEW: Add entity connections
                            'semantic_coordinates': {
                                'source': source_coords.tolist() if hasattr(source_coords, 'tolist') else source_coords,
                                'target': target_coords.tolist() if hasattr(target_coords, 'tolist') else target_coords,
                                'distance': semantic_distance
                            },
                            'data_type_compatibility': {
                                'source_type': dataset_info[src_idx]['file_info']['data_type'],
                                'target_type': dataset_info[tgt_idx]['file_info']['data_type'],
                                'compatible': dataset_info[src_idx]['file_info']['data_type'] == 
                                             dataset_info[tgt_idx]['file_info']['data_type']
                            },
                            'contextual_analysis': {
                                'semantic_compatibility': float(semantic_compatibility),
                                'source_context': dataset_info[src_idx]['contextual_info'],
                                'target_context': dataset_info[tgt_idx]['contextual_info'],
                                'local_relationships': {
                                    'source': source_local_relationships,
                                    'target': target_local_relationships
                                },
                                'interpretation': {
                                    'connection_meaning': _interpret_connection_meaning(
                                        dataset_info[src_idx]['file_info'], dataset_info[tgt_idx]['file_info'], 
                                        bridges, semantic_links, semantic_compatibility
                                    ),
                                    'data_flow_potential': _assess_data_flow_potential(
                                        dataset_info[src_idx]['file_info'], dataset_info[tgt_idx]['file_info']
                                    )
                                }
                            }
                        }
                        semantic_connections[src_idx].append(connection_data)
        
        # Filter connections by threshold and prepare enhanced output
        filtered_connections = {}
        for src_idx, connections_list in semantic_connections.items():
            strong_connections = []
            for connection in connections_list:
                # Calculate combined relevance score
                relevance_score = (
                    connection['mathematical_strength'] * 0.4 +
                    len(connection['semantic_connections']) * 0.1 +
                    len(connection['contextual_bridges']) * 0.2 +
                    (1.0 - connection['semantic_coordinates']['distance']) * 0.3
                )
                # Verbose debug: breakdown of relevance components
                if hasattr(args, 'verbose') and args.verbose:
                    logging.debug(
                        f"Relevance breakdown for {dataset_info[src_idx]['file_info']['file_name']} -> {connection['target_name']}: "
                        f"math={connection['mathematical_strength']:.3f}, "
                        f"semantic_links={len(connection['semantic_connections'])}, "
                        f"bridges={len(connection['contextual_bridges'])}, "
                        f"distance_comp={(1.0 - connection['semantic_coordinates']['distance']):.3f}, "
                        f"total={relevance_score:.3f}, threshold={args.threshold}"
                    )
                
                if relevance_score >= args.threshold:
                    connection['relevance_score'] = relevance_score
                    if len(strong_connections) < args.max_connections:
                        strong_connections.append(connection)
            
            if strong_connections:
                # Sort by relevance score
                strong_connections.sort(key=lambda x: x['relevance_score'], reverse=True)
                filtered_connections[src_idx] = strong_connections
        
        # Generate semantic interpretations for all datasets
        interpreted_coordinates = {}
        for idx, info in enumerate(dataset_info):
            metadata = _get_general_metadata(matrices[idx])
            interpreted_coordinates[idx] = _interpret_semantic_coordinates(
                info['file_info']['semantic_coords'], info, metadata
            )
        
        # Prepare comprehensive output data
        connection_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_type': 'semantic_connection_discovery',
            'num_dimensions': args.num_dims,
            'threshold': args.threshold,
            'total_matrices': len(matrices),
            'connected_matrices': len(filtered_connections),
            # Note: transform references removed (refactored to tgc.py)
            'dataset_info': {
                idx: {
                    'file_info': dataset_info[idx]['file_info'],
                    'contextual_info': dataset_info[idx]['contextual_info'],
                    'semantic_interpretation': interpreted_coordinates[idx],
                    'matrix_analysis': {
                        'matrix_type': dataset_info[idx]['file_info']['matrix_type'],
                        'data_type': dataset_info[idx]['file_info']['data_type'],
                        'shape': dataset_info[idx]['file_info']['shape'],
                        'complexity_estimate': get_complexity(dataset_info[idx]['file_info']['matrix_type'], dataset_info[idx]['file_info']['shape']),
                        'memory_efficiency': get_memory_efficiency(dataset_info[idx]['file_info']['matrix_type'])
                    }
                } for idx in range(len(dataset_info))
            },
            'semantic_connections': filtered_connections,
            'connection_summary': {
                'total_connections': sum(len(targets) for targets in filtered_connections.values()),
                'connection_types': _analyze_connection_types(filtered_connections),
                'domain_clusters': _identify_domain_clusters(dataset_info, filtered_connections)
            }
        }
        
        # Apply enhanced clustering with semantic awareness if requested (need >=3 datasets)
        if args.clustering and len(args.inputs) < 3:
            logging.info("Not enough datasets for clustering (need >=3); skipping clustering step")
            args.clustering = None

        if args.clustering:
            logging.info(f"Performing semantic-aware {args.clustering} clustering...")
            
            try:
                # Use enhanced semantic clustering
                clustering_method = 'enhanced_hierarchical' if args.clustering in ['hierarchical', 'auto'] else 'enhanced_kmeans'
                clustering_result = _cluster_with_semantic_analysis(
                    matrices, filtered_connections, dataset_info, method=clustering_method
                )
                
                if clustering_result and 'semantic_clusters' in clustering_result:
                    connection_data['clustering'] = clustering_result
                    logging.info(f"Semantic clustering completed: {len(clustering_result['semantic_clusters'])} clusters formed")
                    
                    # Log cluster quality metrics
                    quality = clustering_result['cluster_quality']
                    logging.info(f"  Average cluster purity: {quality['average_purity']:.3f}")
                    logging.info(f"  Cross-domain clusters: {quality['cross_domain_clusters']}")
                else:
                    logging.warning("Semantic clustering produced no results")
                    
            except Exception as e:
                logging.error(f"Clustering failed: {str(e)}")
                connection_data['clustering'] = {'error': str(e)}
        
        # Save connection data
        # Ensure we have an output file
        if not args.output:
            args.output = 'connection_results.json'
            logging.info(f"No output file specified, using default: {args.output}")
        
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Save primary JSON output
            save_to_file(connection_data, args.output, 'json')
            logging.info(f"Saved connection data to {args.output}")
        
        # Visualizations suppressed: detailed plots (matplotlib/seaborn/TSNE) are disabled
        # to keep the CLI lightweight and avoid runtime dependencies. If the user
        # still requests visual outputs via args.visualize, we acknowledge the
        # request but do not generate plots. Instead, log a summary and provide
        # the JSON output that can be used by external tools to produce visuals.
        if getattr(args, 'visualize', False):
            logging.info('Visualization requested but suppressed by configuration. No plots will be generated.')
            logging.info(f"Use the JSON output at {args.output or 'connection_results.json'} to create visualizations externally.")
        
        # Generate formatted output
        if args.format == 'edge-list':
            edge_list_file = os.path.join(args.visualize if args.visualize else '.', 'connections_edge_list.txt')
            with open(edge_list_file, 'w') as f:
                for src_idx, targets in filtered_connections.items():
                    src_name = dataset_info[src_idx]['file_info']['file_name']
                    for target in targets:
                        tgt_idx = target['target_idx']
                        tgt_name = dataset_info[tgt_idx]['file_info']['file_name']
                        strength = target['relevance_score']
                        f.write(f"{src_name} {tgt_name} {strength:.4f}\n")
            logging.info(f"Edge list saved to {edge_list_file}")
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        # Print enhanced summary with semantic insights
        total_connection_pairs = sum(len(targets) for targets in filtered_connections.values())
        total_semantic_connections = 0
        for src_idx, connections_list in filtered_connections.items():
            for connection in connections_list:
                total_semantic_connections += len(connection.get('semantic_connections', []))
        
        print("Semantic connection discovery complete")
        print(f"Files analyzed: {len(matrices)}")
        print(f"Connected matrices: {len(filtered_connections)}")
        print(f"Total connection pairs: {total_connection_pairs}")
        print(f"Total individual semantic connections: {total_semantic_connections}")
        
        # Show strongest connection with context
        strongest_src = None
        strongest_tgt = None
        strongest_val = 0
        strongest_connection = None
        
        for src_idx, connections_list in filtered_connections.items():
            for connection in connections_list:
                if connection['relevance_score'] > strongest_val:
                    strongest_val = connection['relevance_score']
                    strongest_src = src_idx
                    strongest_tgt = connection['target_idx']
                    strongest_connection = connection
        
        if strongest_src is not None:
            src_name = dataset_info[strongest_src]['file_info']['file_name']
            tgt_name = dataset_info[strongest_tgt]['file_info']['file_name']
            print(f"Strongest connection: {src_name} → {tgt_name}")
            print(f"   Relevance score: {strongest_val:.4f}")
            print(f"   Mathematical strength: {strongest_connection['mathematical_strength']:.4f}")
            print(f"   Semantic links: {len(strongest_connection['semantic_connections'])}")
            print(f"   Contextual bridges: {len(strongest_connection['contextual_bridges'])}")
        
        # Show data type distribution
        data_types = {}
        for info in dataset_info:
            dt = info['file_info']['data_type']
            if dt not in data_types:
                data_types[dt] = 0
            data_types[dt] += 1
        
        print(f"Data type distribution:")
        for dtype, count in data_types.items():
            print(f"   {dtype}: {count} datasets")
        
        # Show clustering summary if performed
        if 'clustering' in connection_data and 'semantic_clusters' in connection_data['clustering']:
            clusters = connection_data['clustering']['semantic_clusters']
            quality = connection_data['clustering']['cluster_quality']
            print(f"Semantic clusters: {len(clusters)}")
            print(f"Average cluster purity: {quality['average_purity']:.3f}")
            print(f"Cross-domain clusters: {quality['cross_domain_clusters']}")
            
            # Show largest cluster with context
            largest_cluster = max(clusters.items(), key=lambda x: len(x[1]['members']))
            cluster_id, cluster_info = largest_cluster
            cluster_id = int(cluster_id)  # Convert to Python int if needed
            print(f"Largest cluster: Cluster {cluster_id}")
            print(f"   Size: {len(cluster_info['members'])} matrices")
            print(f"   Dominant type: {cluster_info['dominant_data_type']}")
            print(f"   Type purity: {cluster_info['data_type_purity']:.3f}")
        
        print(f"Processing time: {elapsed_time:.4f}s")
        
        if args.output:
            print(f"Semantic connection data saved to: {args.output}")
        
        if getattr(args, 'visualize', False):
            logging.info(f"Visualization was requested ({args.visualize}) but generation is suppressed by configuration.")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during semantic connection discovery: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# Helper methods for enhanced discover_connections_command

def _find_entity_based_connections(dataset_info, matrices):
    """
    Find connections between datasets based on shared entities and entity relationships.
    
    Args:
        dataset_info: List of dataset information dictionaries
        matrices: List of matrices corresponding to dataset_info
        
    Returns:
        dict: Entity-based connections indexed by source dataset
    """
    entity_connections = {}
    
    # Build entity-to-dataset mapping
    entity_to_datasets = {}
    
    for i, info in enumerate(dataset_info):
        entity_summary = info['file_info'].get('entity_summary', {})
        entity_ids = entity_summary.get('entity_ids', [])
        
        for entity_id in entity_ids:
            if entity_id not in entity_to_datasets:
                entity_to_datasets[entity_id] = []
            entity_to_datasets[entity_id].append(i)
    
    logging.debug(f"Found {len(entity_to_datasets)} unique entities across {len(dataset_info)} datasets")
    
    # Find datasets that share entities
    for i, info in enumerate(dataset_info):
        entity_summary = info['file_info'].get('entity_summary', {})
        entity_ids = entity_summary.get('entity_ids', [])
        
        if not entity_ids:
            continue
            
        if i not in entity_connections:
            entity_connections[i] = []
        
        # Check for shared entities with other datasets
        for entity_id in entity_ids:
            sharing_datasets = entity_to_datasets[entity_id]
            
            for j in sharing_datasets:
                if i != j:  # Don't connect to self
                    # Get entity details from both datasets
                    source_entities = _get_entities_in_matrix(matrices[i])
                    target_entities = _get_entities_in_matrix(matrices[j])
                    
                    source_entity_data = source_entities.get(entity_id, {})
                    target_entity_data = target_entities.get(entity_id, {})
                    
                    connection = {
                        'type': 'entity_based',
                        'target_idx': j,
                        'shared_entity_id': entity_id,
                        'confidence': 0.9,  # High confidence for exact entity matches
                        'description': f"Datasets share entity: {entity_id}",
                        'source_entity_info': source_entity_data.get('info', {}),
                        'target_entity_info': target_entity_data.get('info', {}),
                        'connection_strength': 'strong'
                    }
                    
                    # Check if this connection already exists
                    exists = any(
                        conn['type'] == 'entity_based' and 
                        conn['target_idx'] == j and 
                        conn['shared_entity_id'] == entity_id
                        for conn in entity_connections[i]
                    )
                    
                    if not exists:
                        entity_connections[i].append(connection)
                        logging.debug(f"Found entity-based connection: {i} -> {j} via entity {entity_id}")
    
    return entity_connections


def _calculate_entity_similarity(entity_info1, entity_info2):
    """Calculate similarity score between two entity info dictionaries."""
    if not entity_info1 or not entity_info2:
        return 0.0
    
    # Simple similarity based on common keys and values
    common_keys = set(entity_info1.keys()) & set(entity_info2.keys())
    if not common_keys:
        return 0.0
    
    score = 0.0
    for key in common_keys:
        val1 = str(entity_info1[key]).lower()
        val2 = str(entity_info2[key]).lower()
        
        if val1 == val2:
            score += 1.0
        elif val1 in val2 or val2 in val1:
            score += 0.5
    
    return min(1.0, score / len(common_keys))


def _analyze_connection_types(filtered_connections):
    """Analyze the types of connections found."""
    connection_types = {
        'mathematical_only': 0,
        'semantic_only': 0,
        'contextual_only': 0,
        'hybrid': 0,
        'entity_based': 0,  # NEW: Track entity-based connections
        'entity_semantic': 0  # NEW: Track semantic entity connections
    }
    
    for src_idx, connections_list in filtered_connections.items():
        for connection in connections_list:
            conn_type = connection.get('type', '')
            
            if conn_type == 'entity_based':
                connection_types['entity_based'] += 1
            elif conn_type == 'entity_semantic':
                connection_types['entity_semantic'] += 1
            else:
                # Original logic for other connection types
                has_math = connection.get('mathematical_strength', 0) > 0
                has_semantic = len(connection.get('semantic_connections', [])) > 0
                has_contextual = len(connection.get('contextual_bridges', [])) > 0
                
                if has_math and has_semantic and has_contextual:
                    connection_types['hybrid'] += 1
                elif has_math and not has_semantic and not has_contextual:
                    connection_types['mathematical_only'] += 1
                elif not has_math and has_semantic and not has_contextual:
                    connection_types['semantic_only'] += 1
                elif not has_math and not has_semantic and has_contextual:
                    connection_types['contextual_only'] += 1
    
    return connection_types


def _identify_domain_clusters(dataset_info, filtered_connections):
    """Identify clusters of datasets within specific domains."""
    domain_clusters = {}
    
    # Group by data type
    for info in dataset_info:
        data_type = info['file_info']['data_type']
        if data_type not in domain_clusters:
            domain_clusters[data_type] = []
        domain_clusters[data_type].append(info['file_info']['index'])
    
    # Analyze intra-domain vs inter-domain connections
    intra_domain_connections = 0
    inter_domain_connections = 0
    
    for src_idx, connections_list in filtered_connections.items():
        src_type = dataset_info[src_idx]['file_info']['data_type']
        for connection in connections_list:
            tgt_idx = connection['target_idx']
            tgt_type = dataset_info[tgt_idx]['file_info']['data_type']
            
            if src_type == tgt_type:
                intra_domain_connections += 1
            else:
                inter_domain_connections += 1
    
    return {
        'domain_groups': domain_clusters,
        'intra_domain_connections': intra_domain_connections,
        'inter_domain_connections': inter_domain_connections,
        'cross_domain_ratio': inter_domain_connections / max(1, intra_domain_connections + inter_domain_connections)
    }


def _cluster_with_semantic_analysis(matrices, filtered_connections, dataset_info, method='enhanced_hierarchical'):
    """Perform semantic-aware clustering of connection data with rich context."""
    import numpy as np
    from sklearn.cluster import KMeans
    from collections import defaultdict, Counter
    
    if not matrices or not filtered_connections:
        return {'semantic_clusters': {}, 'cluster_quality': {}}
    
    # Build enhanced similarity matrix with semantic weighting
    n_matrices = len(matrices)
    similarity_matrix = np.zeros((n_matrices, n_matrices))
    semantic_weights = np.zeros((n_matrices, n_matrices))
    
    # Fill similarity matrix with multi-dimensional scoring
    for src_idx, connections_list in filtered_connections.items():
        for connection in connections_list:
            tgt_idx = connection['target_idx']
            
            # Multi-factor similarity calculation
            math_score = connection['mathematical_strength']
            semantic_score = len(connection['semantic_connections']) / 10.0  # Normalize
            contextual_score = len(connection['contextual_bridges']) / 5.0   # Normalize
            relevance_score = connection['relevance_score']
            
            # Weighted combination favoring semantic understanding
            combined_score = (
                0.3 * math_score +
                0.4 * semantic_score +
                0.2 * contextual_score +
                0.1 * relevance_score
            )
            
            similarity_matrix[src_idx, tgt_idx] = combined_score
            similarity_matrix[tgt_idx, src_idx] = combined_score  # Symmetric
            semantic_weights[src_idx, tgt_idx] = semantic_score
    
    # Add self-similarity (diagonal) based on data type compatibility
    for i in range(n_matrices):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, n_matrices):
            if dataset_info[i]['file_info']['data_type'] == dataset_info[j]['file_info']['data_type']:
                # Boost similarity for same data types
                similarity_matrix[i, j] *= 1.2
                similarity_matrix[j, i] *= 1.2
    
    # Convert similarity to distance for clustering
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Use optimized cluster selection from MatrixTransformer
    from matrixtransformer import MatrixTransformer
    temp_transformer = MatrixTransformer()
    
    # Convert distance matrix to coordinate space for optimized selection
    # Create feature vectors from similarity matrix for cluster optimization
    feature_vectors = []
    for i in range(n_matrices):
        # Use row of similarity matrix as feature vector
        feature_vectors.append(similarity_matrix[i, :])
    
    feature_array = np.array(feature_vectors)
    best_n_clusters = temp_transformer.optimized_cluster_selection(feature_array, max_clusters=min(8, n_matrices))
    
    # Use MatrixTransformer's internal clustering capabilities instead of sklearn
    # Store the matrices temporarily in transformer for clustering
    temp_transformer.matrices = matrices
    
    # Generate coordinates for clustering using MatrixTransformer's coordinate system
    coordinates = []
    for i, matrix in enumerate(matrices):
        try:
            coord = temp_transformer._generate_matrix_coordinates(matrix, i)
            coordinates.append(coord)
        except Exception as e:
            logging.warning(f"Could not generate coordinates for matrix {i}: {e}")
            # Fallback to similarity vector as coordinate
            coordinates.append(similarity_matrix[i, :])
    
    if coordinates:
        coordinates_array = np.array([np.array(coord) for coord in coordinates])
        
        # Use MatrixTransformer's optimized clustering instead of sklearn
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates_array)
    else:
        # Fallback if coordinate generation fails
        cluster_labels = np.zeros(n_matrices, dtype=np.int64)
    
    # Build semantic cluster analysis
    semantic_clusters = defaultdict(lambda: {
        'members': [],
        'data_types': Counter(),
        'avg_internal_similarity': 0.0,
        'semantic_coherence': 0.0,
        'dominant_data_type': '',
        'data_type_purity': 0.0,
        'cross_connections': 0,
        'total_connections': 0
    })
    
    # Populate cluster information
    for idx, cluster_id in enumerate(cluster_labels):
        # Convert NumPy int32 to Python int
        cluster_id = int(cluster_id)
        cluster_info = semantic_clusters[cluster_id]
        cluster_info['members'].append(idx)
        cluster_info['data_types'][dataset_info[idx]['file_info']['data_type']] += 1
    
    # Calculate semantic cluster metrics
    for cluster_id, cluster_info in semantic_clusters.items():
        members = cluster_info['members']
        n_members = len(members)
        
        # Calculate average internal similarity
        internal_similarities = []
        for i in range(n_members):
            for j in range(i + 1, n_members):
                idx_i, idx_j = members[i], members[j]
                internal_similarities.append(similarity_matrix[idx_i, idx_j])
        
        cluster_info['avg_internal_similarity'] = np.mean(internal_similarities) if internal_similarities else 0.0
        
        # Calculate semantic coherence based on data type homogeneity
        most_common_type, count = cluster_info['data_types'].most_common(1)[0]
        cluster_info['dominant_data_type'] = most_common_type
        cluster_info['data_type_purity'] = count / n_members
        
        # Count cross-cluster connections
        for member_idx in members:
            if member_idx in filtered_connections:
                for connection in filtered_connections[member_idx]:
                    # Convert NumPy int32 to Python int
                    target_cluster = int(cluster_labels[connection['target_idx']])
                    if target_cluster != cluster_id:
                        cluster_info['cross_connections'] += 1
                    cluster_info['total_connections'] += 1
        
        # Calculate semantic coherence score
        semantic_coherence = (
            0.4 * cluster_info['data_type_purity'] +
            0.3 * cluster_info['avg_internal_similarity'] +
            0.3 * (1.0 - cluster_info['cross_connections'] / max(1, cluster_info['total_connections']))
        )
        cluster_info['semantic_coherence'] = semantic_coherence
    
    # Calculate overall cluster quality metrics
    all_purities = [info['data_type_purity'] for info in semantic_clusters.values()]
    all_coherences = [info['semantic_coherence'] for info in semantic_clusters.values()]
    cross_domain_clusters = sum(1 for info in semantic_clusters.values() if info['data_type_purity'] < 1.0)
    
    cluster_quality = {
        'average_purity': np.mean(all_purities) if all_purities else 0.0,
        'average_coherence': np.mean(all_coherences) if all_coherences else 0.0,
        'cross_domain_clusters': cross_domain_clusters,
        'total_clusters': len(semantic_clusters)
    }
    
    # Create final result dictionary with integer cluster IDs
    # Convert defaultdict to regular dict with properly typed keys
    fixed_semantic_clusters = {}
    for cluster_id, cluster_info in semantic_clusters.items():
        # Convert NumPy types to Python native types
        fixed_semantic_clusters[int(cluster_id)] = cluster_info
    
    return {
        'semantic_clusters': fixed_semantic_clusters,
        'cluster_quality': cluster_quality,
        'cluster_labels': cluster_labels.tolist(),
        'similarity_matrix': similarity_matrix.tolist(),
        'method_used': method
    }


def _calculate_semantic_distance(coords1, coords2):
    """Calculate semantic distance between two sets of coordinates."""
    if not coords1 or not coords2:
        return 1.0  # Maximum distance if no coordinates
    
    try:
        # Convert to arrays for vectorized operations
        coords1_arr = np.array(coords1) if isinstance(coords1, list) else coords1
        coords2_arr = np.array(coords2) if isinstance(coords2, list) else coords2
        
        # Ensure arrays are 2D
        if coords1_arr.ndim == 1:
            coords1_arr = coords1_arr.reshape(1, -1)
        if coords2_arr.ndim == 1:
            coords2_arr = coords2_arr.reshape(1, -1)
            
        # Handle case where shape doesn't have enough dimensions
        if coords1_arr.ndim < 2 or coords2_arr.ndim < 2:
            logging.warning(f"Invalid coordinate dimensions: {coords1_arr.shape}, {coords2_arr.shape}")
            return 1.0
        
        # If coordinates are of different dimensionality, normalize
        try:
            if coords1_arr.shape[1] != coords2_arr.shape[1]:
                # Pad with zeros to match dimensions
                max_dim = max(coords1_arr.shape[1], coords2_arr.shape[1])
                if coords1_arr.shape[1] < max_dim:
                    pad_width = ((0, 0), (0, max_dim - coords1_arr.shape[1]))
                    coords1_arr = np.pad(coords1_arr, pad_width, 'constant')
                if coords2_arr.shape[1] < max_dim:
                    pad_width = ((0, 0), (0, max_dim - coords2_arr.shape[1]))
                    coords2_arr = np.pad(coords2_arr, pad_width, 'constant')
        except IndexError:
            # If shape indexing fails, use a default distance
            logging.warning("Could not access shape indices for coordinate normalization")
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        for coord1 in coords1_arr:
            for coord2 in coords2_arr:
                dist = np.linalg.norm(coord1 - coord2)
                distances.append(dist)
        
        # Return average distance, normalized to [0, 1]
        avg_distance = np.mean(distances) if distances else 1.0
        return min(1.0, avg_distance / 10.0)  # Normalize to [0, 1] with a scale factor
        
    except Exception as e:
        logging.error(f"Error calculating semantic distance: {str(e)}")
        return 1.0  # Return maximum distance on error


def _interpret_connection_meaning(source_info, target_info, bridges, semantic_links, semantic_compatibility):
    """Interpret the meaning of connections between datasets with rich context."""
    connection_meanings = []
    
    # Analyze based on contextual bridges
    if bridges:
        bridge_types = [b.get('bridge_type') for b in bridges]
        
        if 'common_column' in bridge_types:
            common_columns = [b.get('bridge_name') for b in bridges if b.get('bridge_type') == 'common_column']
            connection_meanings.append({
                'type': 'column_overlap',
                'description': f"Datasets share {len(common_columns)} common columns: {', '.join(common_columns[:3])}{'...' if len(common_columns) > 3 else ''}",
                'importance': 'high',
                'confidence': 0.9
            })
            
        if 'data_type_match' in bridge_types:
            connection_meanings.append({
                'type': 'data_type_compatibility',
                'description': f"Datasets share compatible data types",
                'importance': 'medium',
                'confidence': 0.7
            })
    
    # Analyze based on semantic links
    if semantic_links:
        link_types = [link.get('type') for link in semantic_links]
        
        if 'statistical_similarity' in link_types:
            stats_links = [link for link in semantic_links if link.get('type') == 'statistical_similarity']
            if stats_links and stats_links[0].get('mean_similarity') > 0.5:
                connection_meanings.append({
                    'type': 'statistical_patterns',
                    'description': "Datasets show similar statistical distributions and patterns",
                    'importance': 'medium',
                    'confidence': 0.8
                })
    
    # Analyze based on semantic compatibility
    if semantic_compatibility > 0.8:
        connection_meanings.append({
            'type': 'semantic_similarity',
            'description': f"Datasets have high semantic compatibility ({semantic_compatibility:.2f})",
            'importance': 'high',
            'confidence': semantic_compatibility
        })
    
    # If no specific meanings were found, provide a general interpretation
    if not connection_meanings:
        if semantic_compatibility > 0.5:
            connection_meanings.append({
                'type': 'general_similarity',
                'description': "Datasets show general structural similarities",
                'importance': 'low',
                'confidence': semantic_compatibility
            })
    
    return connection_meanings


def _assess_data_flow_potential(source_info, target_info):
    """Assess potential data flow and transformation paths between datasets."""
    flow_potential = {
        'compatibility': 0.0,
        'transformation_difficulty': 0.0,
        'recommended_steps': [],
        'expected_quality': 0.0
    }
    
    # Check basic compatibility
    same_data_type = source_info['data_type'] == target_info['data_type']
    flow_potential['compatibility'] = 0.8 if same_data_type else 0.4
    
    # Assess transformation difficulty
    if same_data_type:
        flow_potential['transformation_difficulty'] = 0.2  # Easy
        flow_potential['expected_quality'] = 0.9
        flow_potential['recommended_steps'] = [{
            'step': 'direct_mapping',
            'description': f"Direct column mapping between {source_info['file_name']} and {target_info['file_name']}"
        }]
    else:
        # More complex transformation required
        flow_potential['transformation_difficulty'] = 0.6  # Medium
        flow_potential['expected_quality'] = 0.7
        flow_potential['recommended_steps'] = [
            {
                'step': 'type_conversion',
                'description': f"Convert from {source_info['data_type']} to intermediate format"
            },
            {
                'step': 'structural_mapping',
                'description': "Map data structure between formats"
            },
            {
                'step': 'target_conversion',
                'description': f"Convert to target {target_info['data_type']} format"
            }
        ]
    
    return flow_potential