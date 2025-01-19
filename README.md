# Enhancing E-commerce Search with Elasticsearch KNN: A Practical Implementation

## Introduction

In e-commerce, search functionality is crucial for user experience and conversion rates. Traditional keyword-based searches, while reliable, often fall short when dealing with part numbers, product names, and descriptions that require semantic understanding. This article explores how implementing K-Nearest Neighbors (KNN) search in Elasticsearch can significantly improve search relevance and user experience.

## Project Objectives

This project focused on two main goals:
1. Migrating documents between Elasticsearch servers while incorporating vector embeddings for specific text fields to enable KNN searches
2. Developing a Python-based testing framework for evaluating KNN search capabilities in the new Elasticsearch index

## Understanding KNN Search in E-commerce

### Traditional vs. KNN Search

Traditional e-commerce search relies heavily on exact matches or wildcards. For example, searching for a part number "BABF9885" would only return products with that exact pattern or wildcard matches. Similarly, searching for "Fuel Spin-On" would only match products containing those exact terms.

Let's examine how KNN search transforms this approach:

```json
// Traditional Mapping
{
  "mappings": {
    "properties": {
      "part_number": { "type": "keyword" },
      "product_name": { "type": "text" }
    }
  }
}

// KNN-enabled Mapping with Vectors
{
  "mappings": {
    "properties": {
      "part_number": { "type": "keyword" },
      "product_name": { "type": "text" },
      "product_name_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

### Key Improvements with KNN

1. **Semantic Understanding**
   - Traditional search treats "Spin-On" and "rotating" as entirely different terms
   - KNN recognizes conceptual similarity through vector representations

2. **Fuzzy Part Number Matching**
   - Identifies similar part numbers based on learned patterns
   - Helps customers find products even with slightly incorrect part numbers

3. **Enhanced Product Discovery**
   - Combines traditional and semantic search capabilities
   - Returns both exact and contextually relevant results

## Vectorization Strategies

Text vectorization converts text into numerical vectors through several methods:

### Vector Types

1. **Dense Vectors**
   - Neural network embeddings
   - Learned representations
   - Example: "fuel filter" as 384-dimensional vector: `[0.23, -0.45, 0.12, 0.67, ...]`

2. **Contextual Embeddings**
   - Context-aware representations
   - Different vectors for the same word in different contexts

### Implementation Strategy

For this project, we implemented three distinct vectorization strategies:

1. Part Numbers: 384 dimensions using MiniLM-L12-v2
2. Short Text: 384 dimensions using MiniLM-L3-v2
3. Long Text: 768 dimensions using MPNet-base-v2

## Technical Implementation

### Core Components

1. **Configuration and Setup**
   - Environment-based configuration
   - Comprehensive logging system
   - Parameter validation

2. **Document Processing Pipeline**
   - Scroll API for efficient document reading
   - Configurable batch processing
   - Concurrent processing with multiple workers

3. **Index Management**
   - Automated mapping creation
   - Vector field configuration
   - Optimization for large-scale operations

### Performance Considerations

- Configurable batch sizes
- Multi-threaded processing
- Memory-efficient document streaming
- Real-time progress tracking
- Comprehensive error handling

## Test Results and Validation

Our testing focused on two primary scenarios:

1. **Part Number Search**
   - Search: "BABF9885"
   - Results: Exact match followed by similar patterns (BABF988, BABF989, etc.)
   - Demonstrated effective fuzzy matching capabilities
     ![image](https://github.com/user-attachments/assets/54a7ac94-29e1-49a3-8aaa-f8ac1c99cedd)


2. **Product Description Search**
   - Search: "Fuel rotating"
   - Results: Successfully returned "Fuel Spin-On" products
   - Proved semantic understanding capabilities
     "Fuel spin-on" searches:
   ![image](https://github.com/user-attachments/assets/b3daf819-12f8-42b8-b1b5-eae9fdf3fc1e)

    "Fuel rotating" search matches "Fuel spin-on" products:
  ![image](https://github.com/user-attachments/assets/d5628edd-8077-48c1-aafe-245058618dd0)



## Future Improvements

Future enhancements will focus on:

1. Extending vectorization to longer text fields
2. Implementing token chunking for improved context preservation
3. Optimizing overlap between chunks for better semantic understanding

## Conclusion

This implementation demonstrates the significant advantages of incorporating KNN search in e-commerce applications. The combination of traditional keyword search with vector-based semantic search provides a more robust and user-friendly search experience.

The complete code and detailed test results are available on [GitHub](https://github.com/tezgiden/Elasticsearch-KNN-Search).

---

*Note: This project was developed with assistance from various AI tools, including Claude.ai, demonstrating the transformative impact of AI on modern development cycles.*
