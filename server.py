from mcp.server.fastmcp import FastMCP
import pymongo
from typing import Dict, List, Any, Optional
from bson import ObjectId
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP("mongodb agent", dependencies=["pymongo"]) 

def convert_objectids(obj):
    if isinstance(obj, List):
        return [convert_objectids(item) for item in obj]
    elif isinstance(obj, Dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj


@mcp.tool()
async def get_mongodb_sample_documents(
    database_name: str = "UsersDB",
    collection_name: str = "users",
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get a sample of MongoDB documents from a collection.
    Args:
        database_name (str, optional): The name of the database. Defaults to "UsersDB".
        collection_name (str, optional): The name of the collection to query. Defaults to "users".
        limit (int, optional): The maximum number of documents to return. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of sample documents from the collection.
    """
    client = pymongo.MongoClient(os.environ["MONGODB_CONNECTION_STRING"])
    db = client[database_name]
    collection = db[collection_name]
    samples = list(collection.find({}, limit=limit))
    samples = convert_objectids(samples)
    client.close()
    return samples

@mcp.tool()
async def get_mongodb_databases() -> List[str]:
    """
    Generates a list of MongoDB databases.
    Returns:
        List[str]: A list of MongoDB database names.
    """
    client = pymongo.MongoClient(os.environ["MONGODB_CONNECTION_STRING"])
    return list(client.list_database_names())

@mcp.tool()
async def get_mongodb_collections(database_name: str = "UsersDB") -> List[str]:
    """
    Generates a list of MongoDB collections.
    Returns:
        List[str]: A list of MongoDB collection names.
    """
    client = pymongo.MongoClient(os.environ["MONGODB_CONNECTION_STRING"])
    return list(client[database_name].list_collection_names())

@mcp.tool()
async def execute_mongodb_query(query: Dict[str, Any], 
                                collection_name: str = "users",
                                database_name: str = "UsersDB",
                                connection_string: str = os.environ["MONGODB_CONNECTION_STRING"],
                                projection: Optional[Dict[str, Any]] = None,
                                limit: Optional[int] = None,
                                sort: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
    """Execute a MongoDB query and return the results.
    IMPORTANT: This tool works with MongoDB collections that may contain nested documents.
    Always use dot notation to query nested fields.

    Args:
        query (Dict[str, Any], str): The mongoDB query to execute.
        collection_name (str, optional): The name of the collection to query. Defaults to "users".
        database_name (str, optional): The name of the database. Defaults to "UsersDB".
        connection_string (_type_, optional): MongoDB connection string. Defaults to os.environ["MONGODB_CONNECTION_STRING"].
        projection (Optional[Dict[str, Any]], optional): Fields to include/exclude in the results. Defaults to None.
        limit (Optional[int], optional): Maximum number of results to return. Defaults to None.
        sort (Optional[List[tuple]], optional): List of (key, direction) pairs for sorting. Defaults to None.

    Returns:
        List[Dict[str, Any]]: The query results as a list of dictionaries.
        
    Raises:
        pymongo.errors.ConnectionFailure: If the connection to MongoDB fails.
        pymongo.errors.OperationFailure: If the query operation fails.
        Exception: If any other error occurs.
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(connection_string)
        
        # Access the database 
        db = client[database_name]
        
        # Access the collection
        collection = db[collection_name]
        
        # if type(query) != dict:
        #     query = json.loads(query)
        # Execute the query with optional parameters
        cursor = collection.find(query, projection)
        
        # Apply sort and limits if provided
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert the cursor to a list of dictionaries
        results = list(cursor)
        
        # Object ID conversion for all results
        results = convert_objectids(results)
        
        # Close the connection
        client.close()
        
        return results
    
    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Connection to MongoDB failed: {e}")
    except pymongo.errors.OperationFailure as e:
        raise Exception(f"Query operation failed: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
    
@mcp.tool()
async def get_users_by_city(city: str,
                            projection: Optional[Dict[str, Any]] = None,
                            limit: Optional[int] = None,
                            sort: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
    """
    Gets all the users in a city.

    Args:
        city (str): The name of the city to query.
        projection (Optional[Dict[str, Any]], optional): Fields to include/exclude in the results. Defaults to None.
        limit (Optional[int], optional): Maximum number of results to return. Defaults to None.
        sort (Optional[List[tuple]], optional): List of (key, direction) pairs for sorting. Defaults to None.

    Returns:
        List[Dict[str, Any]]: The query results as a list of dictionaries.
        
    Raises:
        pymongo.errors.ConnectionFailure: If the connection to MongoDB fails.
        pymongo.errors.OperationFailure: If the query operation fails.
        Exception: If any other error occurs.
    """
    try:
        connection_string: str = os.environ["MONGODB_CONNECTION_STRING"]        # Connect to MongoDB
        client = pymongo.MongoClient(connection_string)
        
        collection = client["UsersDB"]["users"]
        
        cursor = collection.find({"address.city": {"$regex": f"{city}", "$options": "i"}}, projection)
        
        # Apply sort and limits if provided
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert the cursor to a list of dictionaries
        results = list(cursor)
        
        # Object ID conversion for all results
        results = convert_objectids(results)
        
        # Close the connection
        client.close()
        
        return results
    
    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Connection to MongoDB failed: {e}")
    except pymongo.errors.OperationFailure as e:
        raise Exception(f"Query operation failed: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
    
@mcp.resource("mongodb://UsersDB/users")
def getuserSchema() -> Dict[str, Any]:
    """
    Get the schema of the "users" collection in MongoDB.
    
    Returns:
        Dict: The schema of the users collection.
    """
    return {
        "_id": {
            "type": "string",
            "required": True,
            "trim": True
        }, 
        "name": {
            "type": str,
            "required": True,
            "trim": True
        },
        "email": {
            "type": str,
            "required": True,
            "trim": True
        },
        "phone": {
            "type": str,
            "required": True,
            "trim": True
        },
        "age": {
            "type": int,
            "required": True,
            "trim": True
        },
        "address": {
            "type": Dict,
            "required": True,
            "trim": True
        },
        "address": {
            "type": Dict,
            "required": True,
            "trim": True,
            "properties": {
                "city": {
                    "type": str,
                    "required": True,
                    "trim": True
                },
                "pincode": {
                    "type": str,
                    "required": True,
                    "trim": True
                }
            }
        }
    }
    
if __name__ == "__main__":
    mcp.run(transport="stdio")
    