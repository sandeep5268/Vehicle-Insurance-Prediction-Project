import os
import sys
import pymongo
import certifi

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """ 
    MongoDBClient is responsible for extablishing a connection to the MongoDB database
    
    Attributes:
    -----------
    client : MongoClient
        A shared MongoClient instance for class.
    database : Database
        The specific database instance that MongoDBClient connect to.
        
    Methods:
    --------
    __init__(database_name: str) -> None
        initializes the MongoDb connection using the given database name.
    """
    
    client = None
    
    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """ 
        Initializes a connnection to the MongoDB database. If no existing connection is found, it establishes a new one.
        """
        try:
            # Check if a mongoDB client connection has already been established; if not , create a new one
            if MongoDBClient.client is None:
                # Retrieve MongoDB URL from environment variables
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")
                
                # Establish a new MongoDD client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
            
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name] # Connects to the specified database
            self.database_name = database_name
            logging.info("MongoDB connection Successful.")
        
        except Exception as e:
            # Raise a custom exception with traceback details if connection fails
            raise MyException(e, sys)