from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from elasticsearch import Elasticsearch
import numpy as np
from sentence_transformers import SentenceTransformer
import dotenv
import os
import logging
from functools import lru_cache
from datetime import datetime
from elasticsearch.exceptions import ConnectionError as ESConnectionError
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class ElasticsearchVectorSearchError(Exception):
    """Base exception for ElasticsearchVectorSearch """
    pass

class ConfigurationError(ElasticsearchVectorSearchError):
    """Raised when configuration is invalid"""
    pass

class ConnectionError(ElasticsearchVectorSearchError):
    """Raised when connection to Elasticsearch fails"""
    pass

class SearchError(ElasticsearchVectorSearchError):
    """Raised when search operation fails"""
    pass

class VectorGenerationError(ElasticsearchVectorSearchError):
    """Raised when vector generation fails"""
    pass

# Constants
@dataclass
class SearchConstants:
    DEFAULT_VECTOR_DIMS: int = 384
    DEFAULT_MIN_SCORE: float = 0.7
    DEFAULT_K: int = 10
    DEFAULT_NUM_CANDIDATES_MULTIPLIER: int = 5
    DEFAULT_MODEL_NAME: str = 'all-MiniLM-L12-v2'
    DEFAULT_CACHE_SIZE: int = 1000

@dataclass
class ElasticsearchConfig:
    """Configuration class for Elasticsearch connection and search parameters"""
    es_url: str
    es_username: str
    es_password: str
    index_name: str
    es_timeout: int = 30
    es_max_retries: int = 3
    es_verify_certs: bool = False
    #connection_pool_size: int = 10

    @classmethod
    def from_env(cls, env_file_path: Optional[str] = None) -> 'ElasticsearchConfig':
        """Create configuration from environment variables"""
        dotenv.load_dotenv(env_file_path)

        required_vars = {
            'es_url': 'TARGET_ES_URL',
            'es_username': 'ES_USERNAME',
            'es_password': 'ES_PASSWORD',
            'index_name': 'TARGET_INDEX'
        }

        config_dict = {}
        for key, env_var in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                raise ConfigurationError(f"Required environment variable {env_var} is not set")
            config_dict[key] = value

        # Optional parameters
        config_dict['es_timeout'] = int(os.getenv('ES_TIMEOUT', '30'))
        config_dict['es_max_retries'] = int(os.getenv('ES_MAX_RETRIES', '3'))
        config_dict['es_verify_certs'] = os.getenv('ES_VERIFY_CERTS', 'False').lower() == 'true'
        #config_dict['connection_pool_size'] = int(os.getenv('ES_CONNECTION_POOL_SIZE', '10'))

        return cls(**config_dict)

class VectorGenerator:
    """Handles vector generation for search terms"""
    def __init__(self, model_name: str = SearchConstants.DEFAULT_MODEL_NAME):
        """Initialize the vector generator with a specific model"""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized VectorGenerator with model: {model_name}")
        except Exception as e:
            raise VectorGenerationError(f"Failed to initialize model: {str(e)}")

    @lru_cache(maxsize=SearchConstants.DEFAULT_CACHE_SIZE)
    def generate_vector(self, search_term: str) -> List[float]:
        """Generate vector for search term with caching"""
        try:
            vector = self.model.encode(search_term).tolist()
            logger.debug(f"Generated vector for search term: {search_term}")
            return vector
        except Exception as e:
            raise VectorGenerationError(f"Failed to generate vector: {str(e)}")

class ElasticsearchVectorSearch:
    """Main class for vector search operations"""
    def __init__(self, config: ElasticsearchConfig, vector_generator: VectorGenerator):
        """Initialize search with configuration and vector generator"""
        self.config = config
        self.vector_generator = vector_generator
        self.es = self._initialize_elasticsearch()
        logger.info(f"Initialized ElasticsearchVectorSearch with index {config.index_name}")

    def _initialize_elasticsearch(self) -> Elasticsearch:
        """Initialize Elasticsearch connection with retry mechanism"""
        try:
            es = Elasticsearch(
                [self.config.es_url],
                basic_auth=(self.config.es_username, self.config.es_password),
                timeout=self.config.es_timeout,
                max_retries=self.config.es_max_retries,
                retry_on_timeout=True,
                verify_certs=self.config.es_verify_certs
                #pool_size=self.config.connection_pool_size
            )

            if not es.ping():
                raise ConnectionError("Failed to connect to Elasticsearch")

            return es
        except ESConnectionError as e:
            raise ConnectionError(f"Failed to initialize Elasticsearch: {str(e)}")

    def _extract_nested_field(self, source: Dict[str, Any], field_path: str) -> Any:
        """Extract nested field value from source dictionary"""
        try:
            current = source
            if isinstance(current, dict) and field_path in current:
                return current[field_path]


            for key in field_path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        except Exception as e:
            logger.warning(f"Failed to extract field {field_path}: {str(e)}")
            return None

    def semantic_search(
        self,
        search_term: str,
        vector_field: str = 'identifier.sku.raw_vector',
        output_fields: Optional[Dict[str, str]] = None,
        k: int = SearchConstants.DEFAULT_K,
        min_score: float = SearchConstants.DEFAULT_MIN_SCORE
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using KNN"""
        try:
            query_vector = self.vector_generator.generate_vector(search_term)
            
            search_query = {
                "size": k,
                "knn": {
                    "field": vector_field,
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": k * SearchConstants.DEFAULT_NUM_CANDIDATES_MULTIPLIER
                },
                "_source": True
            }

            results = self.es.search(
                index=self.config.index_name,
                body=search_query,
                stored_fields="*"
            )

            return self._process_search_results(results, min_score, output_fields)
        except Exception as e:
            raise SearchError(f"Semantic search failed: {str(e)}")

    def multi_vector_search(
        self,
        search_term: str,
        primary_search_field: str = 'identifier.sku.raw_vector',
        secondary_search_field: str = 'custom.sku_normalized_no_special_char_vector',
        output_fields: Optional[Dict[str, str]] = None,
        k: int = SearchConstants.DEFAULT_K,
        min_score: float = SearchConstants.DEFAULT_MIN_SCORE
    ) -> List[Dict[str, Any]]:
        """Perform multi-vector similarity search"""
        try:
            query_vector = self.vector_generator.generate_vector(search_term)
            
            search_query = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "field": primary_search_field,
                                    "query_vector": query_vector,
                                    "k": k,
                                    "num_candidates": k * SearchConstants.DEFAULT_NUM_CANDIDATES_MULTIPLIER
                                }
                            },
                            {
                                "knn": {
                                    "field": secondary_search_field,
                                    "query_vector": query_vector,
                                    "k": k,
                                    "num_candidates": k * SearchConstants.DEFAULT_NUM_CANDIDATES_MULTIPLIER
                                }
                            }
                        ]
                    }
                },
                "_source": True
            }

            results = self.es.search(
                index=self.config.index_name,
                body=search_query,
                stored_fields="*"
            )

            return self._process_search_results(results, min_score, output_fields)
        except Exception as e:
            raise SearchError(f"Multi-vector search failed: {str(e)}")

    def _process_search_results(
        self,
        results: Dict[str, Any],
        min_score: float,
        output_fields: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Process and filter search results"""
        filtered_results = []
        
        for hit in results['hits']['hits']:
            if hit['_score'] >= min_score:
                result = {
                    "id": hit['_id'],
                    "score": hit['_score'],
                    "source": hit.get('_source', {})
                }

                if output_fields:
                    result['output_fields'] = {
                        field_name: self._extract_nested_field(hit['fields'], field_path)
                        for field_name, field_path in output_fields.items()
                    }

                filtered_results.append(result)

        return filtered_results

def main():
    """Example usage of the improved implementation"""
    try:
        # Initialize configuration
        config = ElasticsearchConfig.from_env()
        
        # Initialize vector generator
        vector_generator = VectorGenerator()
        
        # Initialize search
        es_searcher = ElasticsearchVectorSearch(config, vector_generator)
        
        # Example search fields
        output_fields = {
            'identifier_sku': 'identifier.sku.raw',
            'custom_sku': 'custom.sku_normalized_no_special_char',
            "custom_text": "custom.text.normalized_no_special_char",
            "custom_stemmed_search": "custom_stemmed_search",
            "manufacturer.raw": "manufacturer.raw",
            "name.raw":"name.raw"
        }

        # Perform searches: Part Number searches
        search_term = 'BABF9885'
        
        # Single vector search
        single_vector_results = es_searcher.semantic_search(
            search_term=search_term,
            output_fields=output_fields
        )
        
        logger.info("Single Vector Search Results:")
        for result in single_vector_results:
            logger.info(f"ID: {result['id']}")
            logger.info(f"Score: {result['score']}")
            logger.info(f"Fields: {result.get('output_fields', {})}")

        # Multi-vector search
        multi_vector_results = es_searcher.multi_vector_search(
            search_term=search_term,
            output_fields=output_fields
        )
        
        logger.info("\nMulti-Vector Search Results:")
        for result in multi_vector_results:
            logger.info(f"ID: {result['id']}")
            logger.info(f"Score: {result['score']}")
            logger.info(f"Fields: {result.get('output_fields', {})}")

        vector_generator = VectorGenerator( model_name="paraphrase-MiniLM-L3-v2" )
        
        # Initialize search again with the correct Model
        # Name searches
        es_name_searcher = ElasticsearchVectorSearch(config, vector_generator)
        
        search_term = "Fuel Spin-On"

        single_vector_results = es_name_searcher.semantic_search(
            search_term=search_term,
            output_fields=output_fields,
            vector_field="name.raw_vector",
            min_score=0.001
        )

        logger.info("Single Vector Fuel Spin-On Search Results:")
        for result in single_vector_results:
            logger.info(f"ID: {result['id']}")
            logger.info(f"Score: {result['score']}")
            logger.info(f"Fields: {result.get('output_fields', {})}")

        search_term = "Fuel rotating"

        single_vector_results = es_name_searcher.semantic_search(
            search_term=search_term,
            output_fields=output_fields,
            vector_field="name.raw_vector",
            min_score=0.001
        )

        logger.info("Single Vector Fuel rotating Search Results:")
        for result in single_vector_results:
            logger.info(f"ID: {result['id']}")
            logger.info(f"Score: {result['score']}")
            logger.info(f"Fields: {result.get('output_fields', {})}")


    except ElasticsearchVectorSearchError as e:
        logger.error(f"Search failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()