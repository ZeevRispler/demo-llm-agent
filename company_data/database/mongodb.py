from pymongo import MongoClient
from typing import List, Literal
import pandas as pd
import certifi
from langchain_mongodb.vectorstore import MongoDBAtlasVectorSearch


def get_engine(connection_url: str) -> MongoClient:
    """
    Get the MongoDB Connection Client.

    :param connection_url: The MongoDB connection String.

    :return:                   The MongoDB connection String.
    """
    client = MongoClient(connection_url, tlsCAFile=certifi.where())
    return client


def get_items(
    engine: MongoClient,
    kinds: List[Literal["rings", "necklaces", "bracelets", "earrings"]] = None,
    colors: List[str] = None,
    metals: List[str] = None,
    stones: List[str] = None,
    collections: List[str] = None,
    gifts: List[str] = None,
    min_price: float = None,
    max_price: float = None,
    sort_by: Literal["highest_price", "lowest_price", "best_reviews", "best_seller", "newest"] = None,
) -> pd.DataFrame:
    
    db = engine["jewellery"]
    collection = db["products_explode"]

    filter_query = []

    if kinds:
        filter_query += [{"kind": {"$in": kinds}}]
    if colors:
        filter_query += [{"items.colors": {"$in": colors}}]
    if metals:
        filter_query += [{"items.metals": {"$in": metals}}]
    if stones:
        filter_query += [{"items.stones": {"$in": stones}}]
    if collections:
        filter_query += [{"collections": {"$in": collections}}]
    if gifts:
        filter_query += [{"gifts": {"$in": gifts}}]
    if min_price:
        filter_query += [{"items.stocks.price": {"$gte": min_price}}]
    if max_price:
        filter_query += [{"items.stocks.price": {"$lte": max_price}}]


    query = {"$and": filter_query} if len(filter_query)>0 else {}

    print("Query",query)

    sort = None
    if sort_by:
        if sort_by == "highest_price":
            sort = [{"price", -1}]
        elif sort_by == "lowest_price":
            sort = [{"price", 1}]
        elif sort_by == "best_reviews":
            sort = [{"average_rating", -1}]
        elif sort_by == "best_seller":
            sort = [{"total_purchases", -1}]
        elif sort_by == "newest":
            sort = [{"date_added", -1}]

    project = {"$project": {"_id": 0, "stones": "$items.stones", "price": "$items.price", "gifts": 1, "collections": 1,\
                            "colors": "$items.colors","date_added": "$items.date_added",\
                            "metals": "$items.metals", "average_rating": {"$avg": "$reviews.rating"}, "items": 1}}

    pipeline = [{"$match":query}, project]
    if sort:
        pipeline += [{"$sort": sort}]

    items = collection.aggregate(pipeline)

    return pd.DataFrame(items)


def get_user_items_purchases_history(
    engine: MongoClient, user_id: str, last_n_purchases: int = 5
) -> pd.DataFrame:
    db = engine["jewellery"]
    collection = db["purchases_explode"]

    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"date": -1}},
        {"$limit": last_n_purchases},
        {"$project": {"_id": 0, "purchases": 1, "purhcases.stocks": 0, "average_rating": {"$avg": "$reviews.rating"}}},
    ]

    items = list(collection.aggregate(pipeline))
    items = [item["purchases"] for item in items]
    return pd.DataFrame(items)


class MongoDBAtlasVectorSearchIgz(MongoDBAtlasVectorSearch):
    def __init__(self, **kwargs):
        self._connection_string = kwargs.get("connection_string")
        self._namespace = kwargs.get("namespace")
        self._client = MongoClient(self._connection_string, tlsCAFile=certifi.where())
        self._collection = self._client[self._namespace.split(".")[0]][self._namespace.split(".")[1]]
        self._embedding = kwargs.get("embedding")
        self._index_name = kwargs.get("index_name", "vector_index")
        self._text_key = kwargs.get("text_key", "text")
        self._embedding_key = kwargs.get("embedding_key", "embedding")
        self._relevance_score_fn = kwargs.get("relevance_score_fn", "cosine")
        


