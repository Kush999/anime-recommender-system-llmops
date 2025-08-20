from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self,persist_directory:str="chroma_db"):
        try:
            logger.info("Initializing Anime Recommendation Pipeline")
            vector_builder = VectorStoreBuilder(csv_path="data/anime_with_synopsis.csv", persist_directory=persist_directory)
            
            # Try to load existing vector store, build if it doesn't exist
            try:
                retriever = vector_builder.load_vector_store().as_retriever()
                logger.info("Loaded existing vector store")
            except:
                logger.info("Building new vector store...")
                vector_builder.build_and_save_vector_store()
                retriever = vector_builder.load_vector_store().as_retriever()
                logger.info("Vector store built successfully")
            
            self.recommender = AnimeRecommender(retriever,GROQ_API_KEY,MODEL_NAME)
            logger.info("Anime Recommendation Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Anime Recommendation Pipeline: {str(e)}")
            raise CustomException("Error initializing Anime Recommendation Pipeline",e)
        
    def recommend(self,query:str) -> str:
        try:
            logger.info(f"Getting recommendation for query: {query}")
            recommendation = self.recommender.get_recommendation(query)
            logger.info(f"Recommendation: {recommendation}")
            return recommendation
        except Exception as e:
            logger.error(f"Error getting recommendation: {str(e)}")
            raise CustomException("Error getting recommendation",e)