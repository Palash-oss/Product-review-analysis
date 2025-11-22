"""
MongoDB database connection and operations
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from bson import ObjectId


def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj


# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "sentiment_analysis"

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        cls.client = AsyncIOMotorClient(MONGODB_URL)
        try:
            # Test connection
            await cls.client.admin.command('ping')
            print(f"Connected to MongoDB at {MONGODB_URL}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            print("Running without database persistence")
            cls.client = None
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
    
    @classmethod
    def get_database(cls):
        """Get database instance"""
        if cls.client:
            return cls.client[DATABASE_NAME]
        return None


class AnalysisRepository:
    """Repository for sentiment analysis data"""
    
    @staticmethod
    async def save_analysis(analysis_data: Dict[str, Any]) -> Optional[str]:
        """Save analysis to database"""
        db = MongoDB.get_database()
        if db is None:
            return None
        
        collection = db.analyses
        
        # MongoDB can handle datetime objects, but ensure created_at exists
        if 'created_at' not in analysis_data:
            analysis_data['created_at'] = datetime.utcnow()
        # Always update the updated_at timestamp
        analysis_data['updated_at'] = datetime.utcnow()
        
        # Insert document (MongoDB handles datetime serialization internally)
        result = await collection.insert_one(analysis_data)
        return str(result.inserted_id)
    
    @staticmethod
    async def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID"""
        db = MongoDB.get_database()
        if db is None:
            return None
        
        collection = db.analyses
        
        try:
            result = await collection.find_one({"_id": ObjectId(analysis_id)})
            if result:
                result['_id'] = str(result['_id'])
                # Convert datetime objects to ISO strings
                result = serialize_datetime(result)
                return result
        except Exception as e:
            print(f"Error fetching analysis: {e}")
        
        return None
    
    @staticmethod
    async def get_all_analyses(
        skip: int = 0, 
        limit: int = 20,
        sort_by: str = "created_at",
        sort_order: int = -1
    ) -> List[Dict[str, Any]]:
        """Get all analyses with pagination"""
        db = MongoDB.get_database()
        if db is None:
            return []
        
        collection = db.analyses
        
        cursor = collection.find().sort(sort_by, sort_order).skip(skip).limit(limit)
        results = []
        
        async for document in cursor:
            document['_id'] = str(document['_id'])
            # Convert datetime objects to ISO strings
            document = serialize_datetime(document)
            results.append(document)
        
        return results
    
    @staticmethod
    async def search_analyses(
        query: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search analyses by filename or content"""
        db = MongoDB.get_database()
        if db is None:
            return []
        
        collection = db.analyses
        
        # Create text search query
        search_filter = {
            "$or": [
                {"metadata.source_file": {"$regex": query, "$options": "i"}},
                {"data.text": {"$regex": query, "$options": "i"}}
            ]
        }
        
        cursor = collection.find(search_filter).sort("created_at", -1).skip(skip).limit(limit)
        results = []
        
        async for document in cursor:
            document['_id'] = str(document['_id'])
            # Convert datetime objects to ISO strings
            document = serialize_datetime(document)
            results.append(document)
        
        return results
    
    @staticmethod
    async def get_analyses_count() -> int:
        """Get total count of analyses"""
        db = MongoDB.get_database()
        if db is None:
            return 0
        
        collection = db.analyses
        return await collection.count_documents({})
    
    @staticmethod
    async def delete_analysis(analysis_id: str) -> bool:
        """Delete analysis by ID"""
        db = MongoDB.get_database()
        if db is None:
            return False
        
        collection = db.analyses
        
        try:
            result = await collection.delete_one({"_id": ObjectId(analysis_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting analysis: {e}")
            return False
    
    @staticmethod
    async def get_recent_analyses(limit: int = 5) -> List[Dict[str, Any]]:
        """Get most recent analyses"""
        return await AnalysisRepository.get_all_analyses(skip=0, limit=limit)
    
    @staticmethod
    async def get_statistics() -> Dict[str, Any]:
        """Get overall statistics"""
        db = MongoDB.get_database()
        if db is None:
            return {
                'total_analyses': 0,
                'total_entries': 0,
                'avg_sentiment': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        collection = db.analyses
        
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_analyses": {"$sum": 1},
                    "total_entries": {"$sum": "$statistics.total_entries"},
                    "avg_sentiment": {"$avg": "$statistics.average_sentiment"},
                    "total_positive": {"$sum": "$statistics.sentiment_distribution.positive"},
                    "total_neutral": {"$sum": "$statistics.sentiment_distribution.neutral"},
                    "total_negative": {"$sum": "$statistics.sentiment_distribution.negative"}
                }
            }
        ]
        
        result = await collection.aggregate(pipeline).to_list(1)
        
        if result:
            stats = result[0]
            return {
                'total_analyses': stats.get('total_analyses', 0),
                'total_entries': stats.get('total_entries', 0),
                'avg_sentiment': stats.get('avg_sentiment', 0),
                'sentiment_distribution': {
                    'positive': stats.get('total_positive', 0),
                    'neutral': stats.get('total_neutral', 0),
                    'negative': stats.get('total_negative', 0)
                }
            }
        
        return {
            'total_analyses': 0,
            'total_entries': 0,
            'avg_sentiment': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
        }
