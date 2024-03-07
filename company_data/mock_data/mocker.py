import datetime
import json
import os
import random
import uuid
from typing import List, Tuple

import pandas as pd


def _generate_stocks(
    products: List[dict], items: List[dict], min_amount: int = 0, max_amount: int = 5
) -> List[dict]:
    price_map = {"rings": 400, "bracelets": 800, "necklaces": 1000, "earrings": 200}
    size_map = {
        "rings": [3, 5, 7, 9],
        "bracelets": [6.3, 6.7, 7.1, 7.9, 9],
        "necklaces": [16, 17.7, 19.4],
        "earrings": [1],
    }

    stocks = []
    for item in items:
        kind = next(p["kind"] for p in products if item["product_id"] == p["product_id"])
        price = random.randint(1, 10) * 50 + price_map[kind]
        if item["stones"] == "diamonds":
            price *= 3
        if item["metals"] == "white gold":
            price += 200
        for size in size_map[kind]:
            stocks.append(
                {
                    "stock_id": str(uuid.uuid4()).replace("-", ""),
                    "item_id": item["item_id"],
                    "size": size,
                    "amount": random.randint(min_amount, max_amount),
                    "price": price,
                }
            )

    return stocks


def _generate_purchases(
    users: List[dict],
    stocks: List[dict],
    min_date: str,
    max_date: str,
    min_amount: int = 1,
    max_amount: int = 4,
) -> Tuple[List[dict], List[dict]]:
    min_date = datetime.datetime.strptime(min_date, "%m/%d/%Y")
    max_date = datetime.datetime.strptime(max_date, "%m/%d/%Y")

    purchases = []
    for _ in range(10):
        user_id = random.choice(users)["user_id"]
        stock_ids = [
            stock["stock_id"]
            for stock in random.sample(stocks, random.randint(min_amount, max_amount))
        ]
        purchase_id = str(uuid.uuid4()).replace("-", "")
        date = min_date + datetime.timedelta(
            days=random.randint(0, (max_date - min_date).days)
        )
        purchases.append(
            {
                "user_id": user_id,
                "stocks": stock_ids,
                "date": date.strftime("%m/%d/%Y"),
                "purchase_id": purchase_id,
            }
        )

    stock_to_purchase = []
    for p in purchases:
        for sid in p["stocks"]:
            stock_to_purchase.append({"purchase_id": p["purchase_id"], "stock_id": sid})
        p.pop("stocks")

    return purchases, stock_to_purchase


def _generate_reviews(
    stocks: List[dict],
    purchases: List[dict],
    stock_to_purchase: List[dict],
    review_chance: float = 0.5
) -> List[dict]:
    reviews = []
    for stp in stock_to_purchase:
        if random.random() < review_chance:
            continue
        purchase = next(p for p in purchases if p["purchase_id"] == stp["purchase_id"])
        stock = next(s for s in stocks if s["stock_id"] == stp["stock_id"])
        date = datetime.datetime.strptime(purchase["date"], "%m/%d/%Y")
        date += datetime.timedelta(days=random.randint(1, 7))
        rating = random.randint(1, 5)
        review = {
            "review_id": str(uuid.uuid4()).replace("-", ""),
            "item_id": stock["item_id"],
            "user_id": purchase["user_id"],
            "date": date,
            "text": "",  # TODO: Generate a review based on randomize score using ChatGPT
            "rating": rating,
            "is_recommend": rating > 3,
        }
        reviews.append(review)

    return reviews


def generate_mock_data(
    sources_directory: str = "./sources", output_directory: str = "./"
):
    with open(os.path.join(sources_directory, "products.json")) as json_file:
        products = json.load(json_file)
    with open(os.path.join(sources_directory, "items.json")) as json_file:
        items = json.load(json_file)
    with open(os.path.join(sources_directory, "users.json")) as json_file:
        users = json.load(json_file)

    stocks = _generate_stocks(products=products, items=items)
    purchases, stock_to_purchase = _generate_purchases(
        users=users, stocks=stocks, min_date="05/03/2024", max_date="07/03/2024"
    )
    reviews = _generate_reviews(stocks=stocks, purchases=purchases, stock_to_purchase=stock_to_purchase)

    products_df = pd.DataFrame(products)
    items_df = pd.DataFrame(items)
    items_df['date_added'] = pd.to_datetime(items_df['date_added'], format='%m/%d/%Y')
    users_df = pd.DataFrame(users)
    users_df['date_of_birth'] = pd.to_datetime(users_df['date_of_birth'], format='%m/%d/%Y')
    stocks_df = pd.DataFrame(stocks)
    purchases_df = pd.DataFrame(purchases)
    purchases_df['date'] = pd.to_datetime(purchases_df['date'], format='%m/%d/%Y')
    stock_to_purchase_df = pd.DataFrame(stock_to_purchase)
    reviews_df = pd.DataFrame(reviews)
    reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%m/%d/%Y')

    products_df.to_csv(os.path.join(output_directory, "products.csv"), index=False)
    items_df.to_csv(os.path.join(output_directory, "items.csv"), index=False)
    users_df.to_csv(os.path.join(output_directory, "users.csv"), index=False)
    stocks_df.to_csv(os.path.join(output_directory, "stocks.csv"), index=False)
    purchases_df.to_csv(os.path.join(output_directory, "purchases.csv"), index=False)
    stock_to_purchase_df.to_csv(
        os.path.join(output_directory, "stock_to_purchase.csv"), index=False
    )
    reviews_df.to_csv(os.path.join(output_directory, "reviews.csv"), index=False)


if __name__ == "__main__":
    generate_mock_data()
