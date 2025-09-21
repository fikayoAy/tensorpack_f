# Understanding the Results of `traverse_graph_command`

This document explains what happens when you run the graph traversal tool and what information it gives you back. We've written this guide to help both technical and non-technical readers understand the valuable insights this tool provides.

## What This Tool Does

The graph traversal tool helps you explore connections between your datasets in different ways. Think of it as an explorer that can:
- Find the best path between two specific datasets
- Discover which datasets act as "bridges" that connect many others
- Search for specific items (like "Customer ID") across all your datasets

After running, the tool tells you whether it succeeded and saves a detailed report that varies based on which exploration mode you used.

## Common Information in All Modes

No matter which exploration mode you use, your results will always include:

```python
{
    'exploration_type': 'direct_exploration',  # Or 'semantic_bridges' or 'entity_search'
    'applied_transform': 'normalize_l2',       # Any transformations applied (if any)
    'transform_metadata': {...},               # Details about the transformation
}
```

**What this tells you**:
- Which exploration mode was used
- Whether any special data processing was applied
- Basic information about the exploration run

## Mode 1: Dataset Pathway Exploration

This mode shows you the best path between two specific datasets, useful when you want to understand how two files relate to each other.

**What this tells you**:
- How confidently the tool believes these datasets are related
- What elements they share in common
- What transformation steps connect them
- A detailed profile of both the source and target datasets

For example, a pathway exploration result might include:

```python
{
    'pathway_type': 'direct_exploration',
    'applied_transform': 'normalize_l2',   # Any transformation applied to the data
    'transform_metadata': {                # Details about how data was processed
        'normalization_method': 'l2_norm',
        'applied_at': '2024-01-15T14:30:22',
        'preprocessing_steps': ['outlier_removal', 'scaling']
    },
    'confidence': 0.85,                    # Overall confidence in this connection (0.0-1.0)
    'confidence_breakdown': {              # Details about the confidence score
        'structural_validity': 0.90,       # Based on shared entities and structure
        'attention_focus': 0.75,           # How clearly the connection paths are defined
        'weights': {'structural': 0.8, 'attention': 0.1, 'synergy': 0.1},
        'composite_confidence': 0.85       # The weighted final confidence score
    },
    'source_dataset': {
        'file_path': '/path/to/customer_data.csv',
        'file_name': 'customer_data.csv', 
        'index': 0,                        # Position in the dataset collection
        'shape': [1000, 50],               # Dimensions (rows, columns)
        'data_type': 'tabular',            # Type of data (tabular, image, text, json)
        'matrix_type': 'dense',            # Matrix type (dense, sparse, symmetric)
        
        'contextual_info': {               # Rich context about this dataset
            'data_source': 'customer_database',
            'data_format': 'csv',
            'data_type': 'tabular',
            'search_entity': 'Customer ID', # Entity being searched (if applicable)
            'array_shape': [1000, 50],     # Same as shape above
            'total_elements': 50000,       # Total number of data points
            'dimensionality': 2,           # Number of dimensions in the data
            'file_size_bytes': 2097152,    # Size in bytes (2MB)
            'content_type': 'tabular_data',
            'quality_indicators': {        # Data quality metrics
                'has_metadata': true,      # Whether metadata is available
                'metadata_completeness': 0.8, # How complete the metadata is (0-1)
                'interpretability_score': 0.9  # How easily understood (0-1)
            }
        },
        
        'semantic_coordinates': {          # Position in similarity space
            'numerical': [0.42, 0.67, 0.33, 0.51, 0.29, 0.72, 0.44, 0.81, 
                         0.38, 0.65, 0.54, 0.76, 0.59, 0.48, 0.62, 0.51],  # 16D coordinates
            'interpretations': {           # What each dimension means
                'structural_complexity': 0.42,  # How complex the data structure is
                'sparsity_ratio': 0.67,         # How sparse vs. dense
                'symmetry_score': 0.33,         # How symmetrical patterns are
                'value_distribution': 0.51,     # How values are distributed
                'temporal_patterns': 0.29,      # Time-based patterns present
                'hierarchical_structure': 0.72, # Hierarchical organization
                'clustering_tendency': 0.44,    # Natural grouping tendency
                'contextual_coherence': 0.81,   # How coherent the context is
                'data_density': 0.38,           # Density of information
                'pattern_regularity': 0.65,     # How regular patterns are
                'semantic_richness': 0.54,      # Richness of meaning
                'cross_references': 0.76,       # Internal references
                'entity_diversity': 0.59,       # Diversity of entities
                'relationship_strength': 0.48,  # Strength of relationships
                'domain_specificity': 0.62,     # How domain-specific
                'information_entropy': 0.51     # Information entropy measure
            }
        },
        
        'entity_matches': [                # Direct entity matches found
            {
                'entity': 'Customer ID',
                'match_type': 'exact',
                'location': 'column_header',
                'confidence': 0.98,
                'context': 'Primary identifier column'
            },
            {
                'entity': 'Purchase Date',
                'match_type': 'semantic',
                'location': 'column_header',
                'confidence': 0.89,
                'context': 'Transaction timestamp'
            }
        ],
        
        'cross_matches': [                 # Target entities found in source
            {
                'target_entity': 'Client Number',
                'source_match': 'Customer ID',
                'match_type': 'semantic',
                'confidence': 0.91,
                'context': 'Identifier mapping'
            }
        ],
        
        'key_entities': [                  # Most important entities found
            {
                'entity': 'Customer ID',
                'importance': 'high'       # This is a critical element
            },
            {
                'entity': 'Purchase Date',
                'importance': 'medium'     # This is somewhat important
            },
            {
                'entity': 'Product SKU',
                'importance': 'high'
            },
            {
                'entity': 'Transaction Amount',
                'importance': 'high'
            },
            {
                'entity': 'Payment Method',
                'importance': 'medium'
            },
            {
                'entity': 'Shipping Address',
                'importance': 'medium'
            },
            {
                'entity': 'Email',
                'importance': 'high'
            },
            {
                'entity': 'Phone Number',
                'importance': 'medium'
            },
            {
                'entity': 'Customer Type',
                'importance': 'medium'
            },
            {
                'entity': 'Discount Code',
                'importance': 'medium'
            }
        ],
        
        'local_relationships': {           # How elements connect locally
            'significant_elements': [
                {
                    'element': 'Customer ID',
                    'significance': 0.95,
                    'connects_to': ['Name', 'Email', 'Phone Number']
                },
                {
                    'element': 'Transaction Amount',
                    'significance': 0.82,
                    'connects_to': ['Payment Method', 'Discount Code']
                }
            ],
            'contextual_connections': [
                {
                    'source': 'Customer ID',
                    'target': 'Email',
                    'relationship_type': 'identity_link',
                    'strength': 0.89
                },
                {
                    'source': 'Product SKU',
                    'target': 'Transaction Amount',
                    'relationship_type': 'value_dependency',
                    'strength': 0.76
                }
            ],
            'semantic_clusters': [
                {
                    'name': 'Customer Identity',
                    'members': ['Customer ID', 'Email', 'Phone Number'],
                    'coherence': 0.88
                },
                {
                    'name': 'Transaction Details',
                    'members': ['Product SKU', 'Transaction Amount', 'Payment Method'],
                    'coherence': 0.79
                }
            ],
            'data_characteristics': {
                'has_temporal_sequence': true,
                'has_hierarchical_structure': false,
                'entity_relationships': 'strong'
            },
            'summary': {
                'total_elements_found': 25,
                'total_connections': 42,
                'registered_entities_count': 15,
                'search_entity': 'Customer ID',
                'data_type_searched': 'tabular',
                'used_general_analysis': true
            }
        }
    },
    
    'target_dataset': {
        # Same detailed structure as source_dataset, but for the target file
        'file_path': '/path/to/sales_data.xlsx',
        'file_name': 'sales_data.xlsx',
        # ...all other fields with their specific values
    },
    
    'transformation_path': [               # Path between source and target
        {
            'operation': 'projection',
            'parameters': {'dimensions': 16},
            'confidence': 0.92
        },
        {
            'operation': 'normalization',
            'parameters': {'method': 'l2'},
            'confidence': 0.95
        },
        {
            'operation': 'mapping',
            'parameters': {'target_space': 'common_semantic'},
            'confidence': 0.88
        }
    ],
    
    'attention_scores': {                  # How attention is distributed
        'entity_attention': {
            'Customer ID': 0.85,
            'Transaction Amount': 0.72,
            'Product SKU': 0.68
        },
        'path_attention': [0.25, 0.35, 0.40],  # Attention to each path step
        'overall_distribution': 'focused',
        'attention_entropy': 0.72          # Lower means more focused
    },
    
    'semantic_connections': [              # Semantic links between datasets
        {
            'connection_type': 'entity_overlap',
            'entities': ['Customer ID', 'Product SKU'],
            'strength': 0.88,
            'description': 'Strong customer and product identifier overlap'
        },
        {
            'connection_type': 'schema_similarity',
            'matching_elements': 8,
            'strength': 0.75,
            'description': 'Similar data organization patterns'
        }
    ],
    
    'contextual_bridges': [                # Contextual connections
        {
            'bridge_type': 'domain_context',
            'context': 'e-commerce_sales',
            'strength': 0.92,
            'details': 'Both datasets contain e-commerce transaction data'
        },
        {
            'bridge_type': 'temporal_alignment',
            'context': 'overlapping_time_periods',
            'strength': 0.83,
            'details': 'Data covers same time period (Jan-Dec 2023)'
        }
    ],
    
    'contextual_connections': [            # More explicit connections
        {
            'source_element': 'Customer ID',
            'target_element': 'Client Number',
            'connection_type': 'identifier_match',
            'strength': 0.91,
            'evidence': 'Matching patterns and distribution'
        },
        {
            'source_element': 'Transaction Amount',
            'target_element': 'Sale Amount',
            'connection_type': 'value_match',
            'strength': 0.87,
            'evidence': 'Identical numerical values'
        }
    ],
    
    'shared_entities': [                   # Entities found in both datasets
        {
            'entity_name': 'Customer ID',
            'in_source': true,
            'in_target': true,
            'importance': 'high',
            'source_count': 1000,          # Appears 1000 times in source
            'target_count': 1000,          # Appears 1000 times in target
            'confidence': 0.95             # Very confident this is the same entity
        },
        {
            'entity_name': 'Product SKU',
            'in_source': true,
            'in_target': true,
            'importance': 'high',
            'source_count': 2500,
            'target_count': 2500,
            'confidence': 0.97
        },
        {
            'entity_name': 'Transaction Date',
            'in_source': true,
            'in_target': true,
            'importance': 'medium',
            'source_count': 1000,
            'target_count': 1000,
            'confidence': 0.93
        }
    ],
    
    'pathway_metadata': {                  # Additional path information
        'path_info': [
            {
                'step': 1,
                'operation': 'projection',
                'quality_score': 0.92
            },
            {
                'step': 2,
                'operation': 'normalization',
                'quality_score': 0.95
            },
            {
                'step': 3,
                'operation': 'mapping',
                'quality_score': 0.88
            }
        ],
        'matrix_analysis': {
            'source_complexity': 'medium',
            'target_complexity': 'medium',
            'transformation_complexity': 'low',
            'memory_requirements': 'standard'
        },
        'transformation_details': {
            'reversible': true,
            'information_loss': 'minimal',
            'computational_complexity': 'O(n log n)'
        }
    }
}
```

## Mode 2: Semantic Bridges Exploration

This mode finds datasets that act as "bridges" connecting many others. This is useful for discovering central datasets in your collection or finding files that could help integrate different parts of your data ecosystem.

**What this tells you**:
- Which datasets serve as bridges between others
- How strong each bridge is
- What specific connections each bridge makes
- Detailed analysis of the bridge's role in your data ecosystem

For example:

```python
{
    'exploration_type': 'semantic_bridges',
    'bridges_found': 5,                    # Found 5 potential bridge datasets
    'applied_transform': 'dimension_reduction',  # Transform applied to data
    'transform_metadata': {                # Details about the transformation
        'method': 'pca',
        'components': 16,
        'variance_preserved': 0.92
    },
    'bridge_datasets': [                   # List of bridge datasets (up to top 10)
        {
            'bridge_dataset': {            # Information about this bridge dataset
                'file_path': '/path/to/customer_master.csv',
                'file_name': 'customer_master.csv',
                'index': 3,                # Position in dataset collection
                'shape': [5000, 75],       # Dimensions (rows, columns)
                'data_type': 'tabular',    # Type: tabular/image/text/json
                'matrix_type': 'dense'     # Matrix type: dense/sparse/symmetric
            },
            'bridge_score': 0.78,          # Overall bridge connection strength
            'connections': [               # All datasets this connects to
                {
                    'target_dataset': {    # A dataset this connects to
                        'file_name': 'sales_data.xlsx',
                        'file_path': '/path/to/sales_data.xlsx',
                        'data_type': 'tabular',
                        'dataset_index': 2
                    },
                    'connection_strength': 0.85,   # How strong the connection is
                    'pathway': [                   # Path between them
                        {
                            'operation': 'projection',
                            'parameters': {'dimensions': 16},
                            'confidence': 0.91
                        },
                        {
                            'operation': 'mapping',
                            'parameters': {'target_space': 'common_semantic'},
                            'confidence': 0.87
                        }
                    ],
                    'contextual_analysis': {       # Analysis of the connection
                        'bridge_context': {        # Context of bridge dataset
                            'data_source': 'master_database',
                            'data_format': 'csv',
                            'content_type': 'customer_master_records',
                            'temporal_coverage': '2020-2023',
                            'primary_entities': ['Customer', 'Account']
                        },
                        'target_context': {        # Context of target dataset
                            'data_source': 'sales_system',
                            'data_format': 'excel',
                            'content_type': 'transaction_records',
                            'temporal_coverage': '2023',
                            'primary_entities': ['Sale', 'Customer']
                        },
                        'semantic_compatibility': 0.82,  # How compatible they are
                        'semantic_distance': 0.18        # Distance (lower is closer)
                    },
                    'semantic_coordinates': {
                        'bridge_coords': {          # Bridge's position in semantic space
                            'numerical': [0.41, 0.62, 0.35, 0.57, 0.33, 0.69, 0.42, 0.78, 
                                         0.39, 0.61, 0.52, 0.73, 0.58, 0.49, 0.66, 0.55],
                            'interpretations': {     # Meaning of each dimension
                                'structural_complexity': 0.41,
                                'sparsity_ratio': 0.62,
                                'symmetry_score': 0.35,
                                'value_distribution': 0.57,
                                'temporal_patterns': 0.33,
                                'hierarchical_structure': 0.69,
                                'clustering_tendency': 0.42,
                                'contextual_coherence': 0.78,
                                'data_density': 0.39,
                                'pattern_regularity': 0.61,
                                'semantic_richness': 0.52,
                                'cross_references': 0.73,
                                'entity_diversity': 0.58,
                                'relationship_strength': 0.49,
                                'domain_specificity': 0.66,
                                'information_entropy': 0.55
                            }
                        },
                        'target_coords': {          # Target's position in semantic space
                            'numerical': [0.46, 0.59, 0.31, 0.55, 0.38, 0.64, 0.45, 0.72, 
                                         0.41, 0.58, 0.49, 0.68, 0.55, 0.51, 0.63, 0.52],
                            'interpretations': {
                                'structural_complexity': 0.46,
                                'sparsity_ratio': 0.59,
                                'symmetry_score': 0.31,
                                # ... other dimensions ...
                                'information_entropy': 0.52
                            }
                        }
                    },
                    'entity_analysis': {            # Analysis of shared entities
                        'bridge_entities': [        # Top entities in bridge dataset
                            {'entity': 'Customer ID', 'importance': 'high'},
                            {'entity': 'Account Number', 'importance': 'high'},
                            {'entity': 'First Name', 'importance': 'medium'},
                            {'entity': 'Last Name', 'importance': 'medium'},
                            {'entity': 'Email', 'importance': 'high'},
                            # ... up to 25 entities ...
                        ],
                        'target_entities': [        # Top entities in target dataset
                            {'entity': 'Sale ID', 'importance': 'high'},
                            {'entity': 'Customer ID', 'importance': 'high'},
                            {'entity': 'Product SKU', 'importance': 'high'},
                            {'entity': 'Sale Date', 'importance': 'medium'},
                            {'entity': 'Amount', 'importance': 'high'},
                            # ... up to 25 entities ...
                        ],
                        'entity_overlaps': [        # Directly shared entities
                            {
                                'entity_name': 'Customer ID',
                                'context': 'shared_entity',
                                'relevance': 'high'
                            },
                            {
                                'entity_name': 'Email',
                                'context': 'shared_entity',
                                'relevance': 'medium'
                            },
                            {
                                'entity_name': 'Phone Number',
                                'context': 'shared_entity',
                                'relevance': 'low'
                            }
                        ],
                        'cross_entity_matches': [   # Related but not identical entities
                            {
                                'entity_name': 'First Name + Last Name',
                                'source_dataset': 'customer_master.csv',
                                'target_dataset': 'sales_data.xlsx',
                                'matches_found': 985,
                                'confidence': 0.87,
                                'connection_type': 'cross_entity_bridge'
                            },
                            {
                                'entity_name': 'Account Number → Customer ID',
                                'source_dataset': 'customer_master.csv',
                                'target_dataset': 'sales_data.xlsx',
                                'matches_found': 1235,
                                'confidence': 0.79,
                                'connection_type': 'cross_entity_bridge'
                            }
                        ],
                        'total_shared_entities': 8,
                        'total_cross_matches': 12
                    },
                    'semantic_connections': [       # Specific semantic connections
                        {
                            'connection_type': 'entity_based',
                            'strength': 0.87,
                            'description': 'Strong customer identifier links'
                        },
                        {
                            'connection_type': 'schema_based',
                            'strength': 0.72,
                            'description': 'Similar column organization patterns'
                        }
                    ],
                    'contextual_bridges': [         # Contextual connecting elements
                        {
                            'bridge_type': 'domain_overlap',
                            'strength': 0.91,
                            'description': 'Both datasets cover customer domain'
                        },
                        {
                            'bridge_type': 'temporal_overlap',
                            'strength': 0.85,
                            'description': 'Overlapping time periods (2023)'
                        }
                    ],
                    'local_relationships': {        # Local entity relationships 
                        # Similar structure as in pathway mode...
                        'significant_elements': [...],
                        'contextual_connections': [...],
                        'semantic_clusters': [...],
                        'summary': {
                            'total_elements_found': 32,
                            'total_connections': 58
                        }
                    },
                    'attention_scores': {           # Attention analysis
                        'entity_attention': {
                            'Customer ID': 0.89,
                            'Email': 0.76
                            # ... other entities ...
                        },
                        'overall_distribution': 'focused'
                    },
                    'connection_metadata': {        # Additional metadata
                        'data_type_compatibility': true,
                        'matrix_type_compatibility': true,
                        'shape_similarity': 0.68,
                        'connection_quality': 0.82
                    }
                },
                # Additional connections would be listed here...
            ],
            'connectivity_count': 7,       # This bridge connects 7 datasets
            'bridge_analysis': {           # Analysis of this bridge dataset
                'contextual_info': {
                    'data_source': 'master_database',
                    'data_format': 'csv',
                    'content_type': 'customer_master_records',
                    # ... other contextual info ...
                },
                'semantic_coordinates': {
                    'numerical': [0.41, 0.62, 0.35, 0.57, 0.33, 0.69, 0.42, 0.78, 
                                 0.39, 0.61, 0.52, 0.73, 0.58, 0.49, 0.66, 0.55],
                    'interpretations': {
                        'structural_complexity': 0.41,
                        'sparsity_ratio': 0.62,
                        # ... other interpretations ...
                    }
                },
                'key_entities': [          # Most important entities in this bridge
                    {'entity': 'Customer ID', 'importance': 'high'},
                    {'entity': 'Account Number', 'importance': 'high'},
                    {'entity': 'Email', 'importance': 'high'},
                    # ... up to 15 entities ...
                ],
                'local_relationships': {   # Relationships between entities
                    # Similar structure as above...
                },
                'bridge_characteristics': {
                    'data_type': 'tabular',
                    'matrix_type': 'dense',
                    'shape': [5000, 75],
                    'complexity_estimate': 45000.0,
                    'memory_efficiency': 0.82,
                    'centrality_score': 0.76,    # How central in the data ecosystem
                    'diversity_score': 0.68      # How diverse its connections are
                }
            },
            'connection_summary': {        # Summary of all connections
                'total_entity_overlaps': 35,
                'total_cross_matches': 42,
                'total_semantic_connections': 18,
                'total_contextual_bridges': 12,
                'avg_semantic_compatibility': 0.79,
                'data_types_connected': ['tabular', 'json', 'text'],
                'matrix_types_connected': ['dense', 'sparse']
            }
        },
        # Additional bridge datasets would be listed here...
    ],
    'bridge_analysis_summary': {           # Overview of all bridges
        'total_bridges_analyzed': 8,
        'successful_bridges': 5,
        'avg_connections_per_bridge': 4.2,
        'data_type_distribution': {
            'tabular': 5,
            'json': 2,
            'text': 1
        },
        'connectivity_patterns': {         # Patterns found in bridges
            'high_connectivity_bridges': [   # Bridges with many connections
                {
                    'name': 'customer_master.csv',
                    'connections': 7
                },
                {
                    'name': 'product_catalog.json',
                    'connections': 5
                },
                {
                    'name': 'transaction_log.csv',
                    'connections': 4
                }
            ],
            'cross_domain_bridges': [      # Bridges connecting different types
                {
                    'name': 'product_catalog.json',
                    'data_types': ['tabular', 'image', 'text']
                },
                {
                    'name': 'customer_master.csv',
                    'data_types': ['tabular', 'json']
                }
            ],
            'high_compatibility_bridges': [  # Bridges with high compatibility
                {
                    'name': 'customer_master.csv',
                    'compatibility': 0.85
                },
                {
                    'name': 'transaction_log.csv',
                    'compatibility': 0.82
                }
            ]
        }
    }
}
```

## Mode 3: Entity Search Exploration

This mode searches for a specific entity (like "Customer ID" or "Product SKU") across all your datasets. This helps you discover where a particular piece of information appears throughout your data ecosystem.

**What this tells you**:
- Which datasets contain the entity you're looking for
- How relevant each match is
- Context around each match
- Relationships between the matches

For example:

```python
{
    'exploration_type': 'entity_search',
    'search_query': 'Customer ID',         # What you searched for
    'matches_found': 7,                    # Found in 7 datasets
    'applied_transform': 'entity_vectorization',  # Transform applied
    'transform_metadata': {                # Details about the transformation
        'method': 'bert_embeddings',
        'model': 'all-MiniLM-L6-v2',
        'embedding_dimensions': 384
    },
    'results': [                           # Results for each dataset match
        {
            'dataset_index': 0,
            'dataset_name': 'customer_data.csv',
            'dataset_path': '/path/to/customer_data.csv',
            'data_type': 'tabular',
            'entity_matches': [            # Specific matches found
                {
                    'entity': 'Customer ID',
                    'match_type': 'exact',
                    'location': 'column_header',
                    'confidence': 0.98,    # Very confident in this match
                    'context': 'Primary identifier column',
                    'match_details': {
                        'match_location': 'Column A',
                        'value_examples': ['CUST-001', 'CUST-002', 'CUST-003'],
                        'column_position': 0,
                        'cardinality': 'unique',
                        'data_type': 'string',
                        'null_percentage': 0.0
                    }
                },
                {
                    'entity': 'CustomerIdentifier',
                    'match_type': 'semantic',
                    'location': 'derived_field',
                    'confidence': 0.85,    # Good confidence, not exact
                    'context': 'Referenced in formulas',
                    'match_details': {
                        'appears_in': 'Calculated fields',
                        'reference_count': 12,
                        'data_type': 'string',
                        'is_referenced': true
                    }
                },
                {
                    'entity': 'customer_id',
                    'match_type': 'case_variation',
                    'location': 'metadata',
                    'confidence': 0.91,
                    'context': 'Schema definition',
                    'match_details': {
                        'appears_in': 'File metadata',
                        'definition': 'Primary customer identifier'
                    }
                }
            ],
            'semantic_coordinates': {      # Position in similarity space
                'numerical': [0.43, 0.68, 0.32, 0.55, 0.30, 0.71, 0.45, 0.82, 
                             0.39, 0.64, 0.53, 0.74, 0.60, 0.48, 0.61, 0.50],
                'interpretations': {       # What these coordinates mean
                    'structural_complexity': 0.43,
                    'sparsity_ratio': 0.68,
                    'symmetry_score': 0.32,
                    'value_distribution': 0.55,
                    'temporal_patterns': 0.30,
                    'hierarchical_structure': 0.71,
                    'clustering_tendency': 0.45,
                    'contextual_coherence': 0.82,
                    'data_density': 0.39,
                    'pattern_regularity': 0.64,
                    'semantic_richness': 0.53,
                    'cross_references': 0.74,
                    'entity_diversity': 0.60,
                    'relationship_strength': 0.48,
                    'domain_specificity': 0.61,
                    'information_entropy': 0.50
                }
            },
            'contextual_info': {           # Context about this dataset
                'data_source': 'customer_database',
                'data_format': 'csv',
                'data_type': 'tabular',
                'search_entity': 'Customer ID',
                'array_shape': [1000, 50],
                'total_elements': 50000,
                'dimensionality': 2,
                'file_size_bytes': 2097152,
                'content_type': 'tabular_data',
                'quality_indicators': {
                    'has_metadata': true,
                    'metadata_completeness': 0.8,
                    'interpretability_score': 0.9
                }
            },
            'local_relationships': {       # How this entity relates locally
                'significant_elements': [
                    {
                        'element': 'Name',
                        'relationship': 'associated_with',
                        'strength': 0.9,
                        'details': 'Name fields belong to the customer'
                    },
                    {
                        'element': 'Address',
                        'relationship': 'associated_with',
                        'strength': 0.8,
                        'details': 'Address belongs to the customer'
                    },
                    {
                        'element': 'Email',
                        'relationship': 'associated_with',
                        'strength': 0.85,
                        'details': 'Email belongs to the customer'
                    },
                    {
                        'element': 'Purchase History',
                        'relationship': 'one_to_many',
                        'strength': 0.92,
                        'details': 'One customer has many purchases'
                    },
                    {
                        'element': 'Account Status',
                        'relationship': 'property_of',
                        'strength': 0.87,
                        'details': 'Status is a property of the customer'
                    }
                ],
                'contextual_connections': [
                    {
                        'connection': 'Customer ID → Account Number',
                        'relationship_type': 'foreign_key',
                        'strength': 0.88,
                        'description': 'Links to accounts table'
                    },
                    {
                        'connection': 'Customer ID → Orders',
                        'relationship_type': 'primary_key',
                        'strength': 0.93,
                        'description': 'Primary key for customer orders'
                    },
                    {
                        'connection': 'Customer ID → Marketing Segments',
                        'relationship_type': 'grouping_key',
                        'strength': 0.72,
                        'description': 'Used for customer segmentation'
                    }
                ],
                'semantic_clusters': [
                    {
                        'name': 'Identity Fields',
                        'members': ['Customer ID', 'Name', 'Email', 'Phone'],
                        'coherence': 0.88,
                        'description': 'Fields that identify the customer'
                    },
                    {
                        'name': 'Transaction References',
                        'members': ['Customer ID', 'Order ID', 'Invoice Number'],
                        'coherence': 0.81,
                        'description': 'Fields used in transactions'
                    },
                    {
                        'name': 'System References',
                        'members': ['Customer ID', 'Internal Ref', 'Legacy ID'],
                        'coherence': 0.76,
                        'description': 'System-level identification'
                    }
                ],
                'data_characteristics': {
                    'cardinality': 'unique',
                    'format_pattern': 'CUST-\\d{3}',
                    'data_type': 'string',
                    'role': 'primary_key',
                    'distribution': 'uniform',
                    'special_values': ['CUST-000', 'CUST-999'],
                    'constraints': ['not_null', 'unique']
                },
                'summary': {
                    'total_elements_found': 25,
                    'total_connections': 42,
                    'registered_entities_count': 18,
                    'search_entity': 'Customer ID',
                    'data_type_searched': 'tabular',
                    'used_general_analysis': true
                }
            },
            'semantic_context': [          # Path from traversal
                {
                    'operation': 'entity_search',
                    'parameters': {'entity': 'Customer ID', 'confidence_threshold': 0.7},
                    'result_count': 3
                },
                {
                    'operation': 'context_expansion',
                    'parameters': {'depth': 2, 'max_connections': 10},
                    'expanded_entities': 12
                }
            ],
            'relevance_score': 0.98,       # How relevant this result is
            'attention_scores': {          # Analysis of importance
                'entity_attention': {
                    'Customer ID': 0.92,
                    'CustomerIdentifier': 0.76,
                    'customer_id': 0.81
                },
                'context_attention': {
                    'primary_key': 0.88,
                    'identifier': 0.82,
                    'customer_data': 0.75
                },
                'overall_distribution': 'highly_focused',
                'attention_entropy': 0.42  # Low entropy = very focused
            },
            'traverse_metadata': {         # Additional traversal metadata
                'path_complexity': 'low',
                'match_confidence': 'high',
                'search_efficiency': 0.94,
                'entity_coverage': 0.98,
                'performance_metrics': {
                    'search_time_ms': 120,
                    'context_expansion_time_ms': 250,
                    'total_processing_time_ms': 370
                }
            },
            'dataset_info_full': {        # Complete dataset information
                'file_info': {
                    'file_path': '/path/to/customer_data.csv',
                    'file_name': 'customer_data.csv',
                    'file_extension': 'csv',
                    'file_size_bytes': 2097152,
                    'creation_date': '2023-05-15T10:30:00Z',
                    'last_modified': '2023-09-22T14:45:22Z',
                    'encoding': 'UTF-8',
                    'has_header': true,
                    'delimiter': ','
                },
                'schema': {
                    'columns': 50,
                    'rows': 1000,
                    'column_names': ['Customer ID', 'First Name', 'Last Name', ...],
                    'column_types': ['string', 'string', 'string', ...],
                    'primary_keys': ['Customer ID'],
                    'foreign_keys': [
                        {'column': 'Account ID', 'references': 'accounts.Account ID'}
                    ]
                },
                'quality_metrics': {
                    'completeness': 0.96,
                    'uniqueness': 0.98,
                    'consistency': 0.92,
                    'timeliness': 0.85
                }
            }
        },
        # Additional results would be listed here...
    ]
}
```

## File Export Options

The tool saves your results in multiple formats:

1. **JSON File** - The complete results with all details
2. **Numeric Files** - Technical arrays for further analysis (.npy or .npz files)
3. **Additional Formats** (optional):
   - CSV spreadsheets for tabular views
   - Excel workbooks with multiple sheets
   - HTML web pages for interactive viewing
   - Database files for running your own queries
   - Parquet files for big data analysis
   - Markdown documentation for sharing

## Visual Output

The tool also provides immediate visual feedback in the console:

1. **Pathway Tables** - Shows connections between source and target datasets
2. **Bridge Summaries** - Ranks bridge datasets by their connectivity
3. **Pathway Trees** - Shows hierarchical relationships between entities

## Error Handling

If something goes wrong, the tool provides a structured error response:

```python
{
    'error': 'Failed to process dataset: file not found',
    'exploration_type': 'error',
    'details': {
        'file_path': '/path/to/missing_file.csv',
        'error_type': 'FileNotFoundError'
    }
}
```

## What Makes These Results Valuable

The graph traversal tool provides several unique insights:

### **Confidence Scoring**
The tool calculates confidence scores to tell you how reliable each finding is:
- **Structural validity** (80%) - Based on shared entities and hard evidence
- **Attention focus** (10%) - How clearly the connection paths are defined
- **Synergy factors** (10%) - How well everything fits together

### **Multiple Exploration Modes**
Different modes let you answer different questions:
- **Pathway exploration** - "How are these two datasets connected?"
- **Bridge discovery** - "Which datasets are central to my data ecosystem?"
- **Entity search** - "Where does this specific piece of information appear?"

### **Rich Entity Analysis**
The tool goes beyond simple matches:
- It finds both exact matches and semantically similar entities
- It maps relationships between entities within datasets
- It calculates confidence for each match

### **Visualization Options**
Results are presented visually in multiple ways:
- Command-line tables and trees for immediate feedback
- Multiple export formats for further analysis
- Detailed exploration paths to show the route between datasets

These results help both technical and business users understand the pathways between datasets, find central connection points, and locate specific information across their data ecosystem.