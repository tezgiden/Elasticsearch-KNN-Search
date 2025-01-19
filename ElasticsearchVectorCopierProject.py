import os
import sys
import time
import logging
import concurrent.futures
from typing import Iterator, Dict, List, Any, Optional
from dataclasses import dataclass, field

import dotenv
from tqdm import tqdm
from retry import retry
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk, BulkIndexError

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='C:/Users/tezgi/Documents/Columbia/elasticsearch-8.15.3/TP-VectorSearch/logs/es_copy_operation.logs', level=logging.INFO):
    """
    Set up comprehensive logging with rotation and multiple handlers
    
    Args:
        log_file: Path to the log file
        level: Logging level 
    """
    # Ensure log directory exists
       # Ensure log directory exists (get directory path from log file path)
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('elasticsearch_copier')
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File Handler with rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger
logger = setup_logging()

@dataclass
class VectorStrategy:
    """
    Configuration for vector embedding strategy
    
    Attributes:
        field_type: Type of text field
        dimensions: Vector embedding dimensions
        model_name: Name of the sentence transformer model
        model: Sentence transformer model instance
        example_fields: Example fields using this strategy
    """
    field_type: str
    dimensions: int
    model_name: str
    model: SentenceTransformer
    example_fields: List[str]

@dataclass
class IndexConfig:
    """
    Configuration for Elasticsearch index copying and vector generation
    
    Attributes:
        source_index: Source Elasticsearch index name
        target_index: Target Elasticsearch index name
        batch_size: Number of documents to process in a single batch
        scroll_size: Number of documents to retrieve in a single scroll
        scroll_timeout: Timeout for Elasticsearch scroll
        vector_dimensions: Default vector dimensions
        num_workers: Number of concurrent workers for processing
        text_fields: Fields to generate vectors for
    """
    source_index: str
    target_index: str
    batch_size: int = 1000
    scroll_size: int = 5000
    scroll_timeout: str = "15m"
    vector_dimensions: int = 384
    num_workers: int = 1
    text_fields: List[str] = field(default_factory=lambda: [
        'identifier.sku.raw', 'custom.sku_normalized_no_special_char', 'default.mpn.raw', 'name.raw', 'manufacturer.raw','custom_stemmed_search', 'custom.text.normalized_no_special_char'
    ])

    def validate(self):
        """
        Validate configuration parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.source_index or not self.target_index:
            raise ValueError("Source and target indices must be specified")
        
        if self.batch_size <= 0 or self.scroll_size <= 0:
            raise ValueError("Batch and scroll sizes must be positive")

class ElasticsearchVectorCopier:
    def __init__(
        self, 
        source_es: Elasticsearch, 
        target_es: Elasticsearch, 
        config: IndexConfig
    ):
        """
        Initialize Elasticsearch index copier with vector generation
        
        Args:
            source_es: Source Elasticsearch client
            target_es: Target Elasticsearch client
            config: Index configuration
        """
        self.source_es = source_es
        self.target_es = target_es
        self.config = config
        
        # Initialize vector strategies
        self.strategies = self._initialize_vector_strategies()
        
        # Initialize statistics tracking
        self.stats = {
            'processed': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }

    def _initialize_vector_strategies(self) -> Dict[str, VectorStrategy]:
        """
        Initialize vector embedding strategies
        
        Returns:
            Dictionary of vector strategies

        We use 3 different models to generate the embeddings based on the anticipated field length:
            - all-MiniLM-L6-v2 : is a lightweight transformer model with 6 layers, offering a good balance between performance and speed, producing 384-dimensional embeddings that work well for semantic similarity tasks and requires less computational resources than its larger variants.
            - all-MiniLM-L12-v2 : represents an enhanced version with 12 layers, providing better accuracy while maintaining reasonable inference speed, also producing 384-dimensional embeddings but with improved semantic understanding due to its deeper architecture.
            - all-mpnet-base-v2 : is based on the MPNet architecture and stands as the most powerful model among the three, trained with masked and permuted language modeling objectives, generating 768-dimensional embeddings that excel in capturing complex semantic relationships and achieving state-of-the-art performance on various benchmarks.
            All three models are trained on diverse datasets including MSMarco, Wikipedia, and CommonCrawl, making them versatile for various text similarity tasks, sentence classification, and information retrieval applications.
            These models support multiple languages and are optimized for sentence-level embeddings, though they can handle texts up to 512 tokens in length, with all-mpnet-base-v2 generally providing the best results at the cost of higher computational requirements.

            There are many others available now and will be available in the future. Pick the ones you feel comfortable.
        """
        return {
            'part_numbers': VectorStrategy(
                field_type='part_numbers',
                dimensions=384,
                model_name='all-MiniLM-L12-v2',
                model=SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2'), #You may want to swap this with all-MiniLM-L6-v2. Makesure that you use the same transformer when query the data
                example_fields=[
                    'identifier.sku.raw', 
                    'custom.sku_normalized_no_special_char', 
                    'default.mpn.normalized'
                ]
            ),
            'short_text': VectorStrategy(
                field_type='short_text',
                dimensions=384,
                model_name='paraphrase-MiniLM-L3-v2',
                model=SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2'),#You may want to swap this with all-MiniLM-L12-v2. Makesure that you use the same transformer when query the data
                example_fields=['name.raw', 'manufacturer.raw']
            ),
            'long_text': VectorStrategy(
                field_type='long_text',
                dimensions=768,
                model_name='all-mpnet-base-v2',
                model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
                example_fields=[
                    'custom_stemmed_search', 
                    'custom.text.normalized_no_special_char'
                ]
            )
        }

    def get_vector_strategy(self, field: str) -> Optional[VectorStrategy]:
        """
        Dynamically select vector strategy based on text characteristics
        
        Args:
            text: Input text to determine strategy
        
        Returns:
            Appropriate VectorStrategy or None
        """

        for strategy in self.strategies.values():
            if field in strategy.example_fields:
                return strategy
        
        return self.strategies['long_text']

    def generate_vectors(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate vector embeddings for specified fields
        
        Args:
            doc: Source document
        
        Returns:
            Document with added vector fields
        """
        try:
            for field in self.config.text_fields:
                try:
                    field_value = self.get_nested_field(doc, field)
                    
                    if not field_value or not isinstance(field_value, (str, list)):
                        continue
                    
                    # Convert list to string if necessary
                    if isinstance(field_value, list):
                        field_value = ' '.join(str(v) for v in field_value)
                    
                    # Select appropriate strategy
                    strategy = self.get_vector_strategy(field)
                    if strategy:
                        doc[f"{field}_vector"] = strategy.model.encode(field_value).tolist()
                
                except Exception as e:
                    logger.warning(f"Error processing field {field}: {e}")
            
            return doc
        
        except Exception as e:
            logger.error(f"Unexpected error in vector generation: {e}")
            return doc

    def get_nested_field(self, doc: Dict[str, Any], field_path: str) -> Optional[Any]:
        """
        Retrieve a nested field from a dictionary using dot notation.

        For example identifier.mpn.raw is stored in following JSON format:
        'identifier': {
            'specification': 'product', 
            'language': 'en_US', 
            'mpn': {
                'normalized': 'BF9885', 
                'raw': 'BF9885'
            }, 
            'sku': {
                'parent': 'P_BABF9885', 
                'normalized': 'BABF9885', 
                'raw': 'BABF9885'
            }
        }

        In this case we need to search for each level individual to find the value we are looking for.
        
        Args:
            doc: Source dictionary
            field_path: Dot-separated path to the field
        
        Returns:
            Field value or None if not found
        """
        keys = field_path.split('.')
        current = doc
        if current.get(field_path):
            logger.info(f"get_nested_field found field_path {field_path} in doc {current}") 
            return current.get(field_path)  
        else:
            logger.info(f"get_nested_field did not find field_path {field_path} in doc {current}")  

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
                
                if current is None:
                    return None
            
            return current
        except Exception as e:
            logger.error(f"Error retrieving nested field {field_path}: {e}")
            return None

    def create_target_index(self) -> None:
        """
        Create target index with vector field mappings
        
        Uses source index settings and adds vector field mappings

        Depending on your data, you might have a huge set of mappings and there is a limit of settings you can define. In our code I increase this limitation to 30000 but it may not be enough in your case.
        Instead of copying the setting from Elasticsearch which might have all the mappings created by the ES based on the data, you may want to use the Base Schema from your ES definition. that will make the mapping section easier to implement.
        """
        try:
            # Get the source index settings and mappings
            source_index_data = self.source_es.indices.get(index=self.config.source_index)[self.config.source_index]
            
            # Extract settings and mappings
            source_settings = source_index_data.get('settings', {}).get('index', {})
            source_mappings = source_index_data.get('mappings', {})
            
            # Add vector field mappings for each strategy
            for strategy in self.strategies.values():
                for field in strategy.example_fields:
                    source_mappings["properties"][f"{field}_vector"] = {
                        "type": "dense_vector",
                        "dims": strategy.dimensions,
                        "index": True,
                        "similarity": "cosine"
                    }
            
            # Prepare clean settings for index creation
            clean_settings = {}
            max_fields=30000

            # Whitelist of setting keys to keep. Others we will not move to our new index. 
            allowed_settings = [
                'number_of_shards',
                'number_of_replicas',
                'analysis',
                'max_ngram_diff',
                'max_shingle_diff',
                'blocks',
                'refresh_interval',
                'routing',
                'search',
                'indexing',
            ]
            
            # Filter settings
            for key in allowed_settings:
                if key in source_settings:
                    clean_settings[key] = source_settings[key]
            # Add field limit settings
            clean_settings['mapping'] = {
                'total_fields': {
                    'limit': max_fields
                }
            }
            
            # Add nested field limit settings
            clean_settings['mapping']['nested_fields'] = {
                'limit': max_fields
            }
            
            # Prepare creation request
            creation_settings = {
                'settings': clean_settings,
                'mappings': source_mappings
            }
            
            
            logger.info(f"create_target_index Done with strategy loop creation_settings ={creation_settings} with vector mappings")
            
            if self.target_es.indices.exists(index=self.config.target_index):
                logger.warning(f"Target index {self.config.target_index} already exists. Deleting...")
                self.target_es.indices.delete(index=self.config.target_index)
            logger.info(f"create_target_index creating the index {self.config.target_index} with vector mappings")
            
            self.target_es.indices.create(
                index=self.config.target_index,
                body=creation_settings
            )
            logger.info(f"Created target index {self.config.target_index} with vector mappings")
        
        except Exception as e:
            logger.error(f"Error creating target index: {e}")
            raise

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents with vector generation.
        You may increase the num_workers to speed up this process depending on your hardware
        
        Args:
            batch: List of source documents
        
        Returns:
            List of processed documents ready for indexing
        """
        processed_docs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(self.generate_vectors, doc) for doc in batch]
            for future in concurrent.futures.as_completed(futures):
                try:
                    processed_doc = future.result()
                    processed_docs.append({
                        '_index': self.config.target_index,
                        '_source': processed_doc
                    })
                except Exception as e:
                    logger.error(f"Error processing document in batch: {e}")
                    self.stats['failed'] += 1
        return processed_docs

    def copy_index(self) -> None:
        """
        Main method to copy index with progress tracking and vector generation
        """
        try:
            # Track start time
            self.stats['start_time'] = time.time()
            
            # Create target index
            self.create_target_index()
            
            # Get total document count
            total_docs = self.source_es.count(index=self.config.source_index)['count']
            logger.info(f"Starting copy operation for {total_docs:,} documents")
            
            # Initialize progress bar
            with tqdm(total=total_docs, desc="Copying documents") as pbar:
                # Process documents in batches
                batch = []
                for doc in self._get_source_documents():
                    batch.append(doc)
                    
                    if len(batch) >= self.config.batch_size:
                        processed_batch = self.process_batch(batch)
                        self._bulk_index_batch(processed_batch)
                        
                        # Update progress
                        pbar.update(len(batch))
                        batch = []
                        
                        # Log progress periodically
                        if self.stats['processed'] % 10000 == 0:
                            self._log_progress()
                
                # Process remaining documents
                if batch:
                    processed_batch = self.process_batch(batch)
                    self._bulk_index_batch(processed_batch)
                    pbar.update(len(batch))
            
            # Track end time and log final statistics
            self.stats['end_time'] = time.time()
            self._log_final_stats()
        
        except Exception as e:
            logger.error(f"Error in copy operation: {e}")
            raise
        finally:
            # Refresh index
            self.target_es.indices.refresh(index=self.config.target_index)

    def _get_source_documents(self) -> Iterator[Dict[str, Any]]:
        """
        Get documents from source index using scroll API
        
        Yields:
            Source documents
        """
        try:
            # This is the place to target sub-set of your catalog. I copy everything from the source as default
            query = {
                "query": {"match_all": {}},
                "_source": ["*"],
                "stored_fields": ["*"],
                "size": self.config.scroll_size
            }

            for doc in scan(
                client=self.source_es,
                index=self.config.source_index,
                query=query,
                scroll=self.config.scroll_timeout,
                size=self.config.scroll_size
            ):
                yield doc['_source']
        except Exception as e:
            logger.error(f"Error scanning source documents: {e}")
            raise

    def _bulk_index_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Bulk index a batch of documents with retry logic
        
        Args:
            batch: List of documents to index
        """
        try:
            success, failed = bulk(
                client=self.target_es,
                actions=batch,
                chunk_size=self.config.batch_size,
                raise_on_error=True  
            )
            
            # Log details about failed documents
            if failed:
                for error in failed:
                    action = error.get('index', error.get('update', {}))
                    error_type = error.get('error', {}).get('type', 'Unknown Error')
                    error_reason = error.get('error', {}).get('reason', 'No reason provided')
                    
                    logger.error(f"Failed document: {action}")
                    logger.error(f"Error Type: {error_type}")
                    logger.error(f"Error Reason: {error_reason}")

            self.stats['processed'] += success
            self.stats['failed'] += len(failed) if failed else 0
        except BulkIndexError as e:
            logger.error(f"Bulk indexing error: {str(e)}")
            raise

    def _log_progress(self) -> None:
        """Log intermediate progress statistics"""
        elapsed_time = time.time() - self.stats['start_time']
        docs_per_second = self.stats['processed'] / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Progress: {self.stats['processed']:,} documents processed "
            f"({docs_per_second:.2f} docs/sec), "
            f"{self.stats['failed']} failed"
        )

    def _log_final_stats(self) -> None:
        """Log final copy operation statistics"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        docs_per_second = self.stats['processed'] / total_time if total_time > 0 else 0
        
        logger.info(
            f"\nCopy operation completed:\n"
            f"Total documents processed: {self.stats['processed']:,}\n"
            f"Failed documents: {self.stats['failed']}\n"
            f"Total time: {total_time:.2f} seconds\n"
            f"Average speed: {docs_per_second:.2f} docs/sec"
        )

def load_elasticsearch_config():
    """
    Load Elasticsearch configuration from environment.
    You need to customize the .env file which is the part of this project
    
    Returns:
        Source and target Elasticsearch clients
    """
    # Load environment variables
    dotenv.load_dotenv()
    
    # Source Elasticsearch configuration
    source_es = Elasticsearch(
        [os.getenv('SOURCE_ES_URL', 'http://localhost:9211')],
        timeout=int(os.getenv('ES_TIMEOUT', 30)),
        max_retries=int(os.getenv('ES_MAX_RETRIES', 3)),
        retry_on_timeout=True
    )
    
    # Target Elasticsearch configuration
    target_es = Elasticsearch(
        [os.getenv('TARGET_ES_URL', 'https://localhost:9204')],
        basic_auth=(
            os.getenv('ES_USERNAME', 'elastic'), 
            os.getenv('ES_PASSWORD')
        ),
        timeout=int(os.getenv('ES_TIMEOUT', 30)),
        max_retries=int(os.getenv('ES_MAX_RETRIES', 3)),
        retry_on_timeout=True,
        verify_certs=os.getenv('ES_VERIFY_CERTS', 'False').lower() == 'true'
    )
    
    return source_es, target_es

def main():
    """
    Main execution function for Elasticsearch index copying
    """
    try:
        # Load Elasticsearch configuration
        source_es, target_es = load_elasticsearch_config()
        
        # Configure index copy parameters
        config = IndexConfig(
            source_index=os.getenv('SOURCE_INDEX'),
            target_index=os.getenv('TARGET_INDEX'),
            batch_size=int(os.getenv('BATCH_SIZE', 10)),
            scroll_size=int(os.getenv('SCROLL_SIZE', 50)),
            num_workers=int(os.getenv('NUM_WORKERS', 1))
        )
        
        # Validate configuration
        config.validate()
        
        # Initialize copier
        copier = ElasticsearchVectorCopier(
            source_es=source_es,
            target_es=target_es,
            config=config
        )
        
        # Perform index copy
        copier.copy_index()
    
    except Exception as e:
        logger.error(f"Copy operation failed: {e}")
        sys.exit(1)
    finally:
        # Close Elasticsearch connections
        source_es.close()
        target_es.close()

if __name__ == "__main__":
    main()